"""事件驱动回测引擎。

每 0.5 秒一个步长，单日 32400 步。单步时序：
1. 市场演化（订单流 -> 中间价 -> 订单簿迁移）
2. 我方上一步入场的挂单做成交判定（含被价格穿越的立即成交）
3. 策略重报价（AS 公式）-> 风控过滤 -> 进入下一步入场
4. 尾盘清仓 / 收盘强平 / 亏损熔断

记账恒等式（每步校验）：equity = cash + q * multiplier * mid。
跨模式基准对比通过 [seed, day] 派生每日随机流保证市场路径一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    BacktestParams,
    MarketParams,
    RiskParams,
    StrategyParams,
    TRADING_SECONDS_PER_DAY,
)
from .contracts import ContractSpec
from .market_sim import MarketSimulator
from .risk import PHASE_FLATTEN, PHASE_TRADING, RiskManager, SessionSchedule
from .strategy import ASStrategy, MODE_AS


@dataclass
class Fill:
    """一笔成交记录。"""

    day: int
    ts: float          # 交易秒
    step: int
    side: int          # +1 买入 / -1 卖出
    price: float
    lots: float
    fee: float
    mid: float
    note: str = "mm"   # mm=做市成交 / liquidation=清仓 / force=强平 / halt=熔断平仓
    rel: str = ""      # 成交渠道：marketable / improve / touch


@dataclass
class QuoteEpisode:
    """一段挂单暴露记录（参数校准用）。"""

    day: int
    side: str          # 'bid' / 'ask'
    delta: float       # 距中间价距离（价格单位，正=远离）
    exposure: float    # 暴露时长（秒）
    filled: bool


@dataclass
class BacktestResult:
    """回测结果容器。"""

    contract: ContractSpec
    mode: str
    market_params: MarketParams
    strategy_params: StrategyParams
    risk_params: RiskParams
    n_days: int
    fills: list[Fill] = field(default_factory=list)
    daily_pnl: list[float] = field(default_factory=list)
    quote_episodes: list[QuoteEpisode] = field(default_factory=list)
    halts: list[tuple[int, str]] = field(default_factory=list)
    # 等间隔采样（sample_every 步）供绘图
    sample_ts: list[float] = field(default_factory=list)
    sample_equity: list[float] = field(default_factory=list)
    sample_inventory: list[int] = field(default_factory=list)
    sample_mid: list[float] = field(default_factory=list)
    sample_bid: list[float] = field(default_factory=list)
    sample_ask: list[float] = field(default_factory=list)
    sample_bb: list[float] = field(default_factory=list)
    sample_ba: list[float] = field(default_factory=list)
    # 每日全分辨率中间价（markout 计算）
    mid_full_days: list[list[float]] = field(default_factory=list)
    # 汇总指标（analytics 填充）
    metrics: dict = field(default_factory=dict)


class PositionTracker:
    """持仓记账：均价法核算已实现盈亏，cash 含手续费。"""

    def __init__(self, contract: ContractSpec):
        self.contract = contract
        self.q = 0.0
        self.cash = 0.0
        self.fees = 0.0
        self.realized_gross = 0.0   # 已实现毛利（不含费用）
        self.closed_lots = 0.0      # 已平手数（每手计一次回合）
        self.n_buys = 0
        self.n_sells = 0
        self.avg_cost = 0.0

    def apply(self, side: int, price: float, lots: float, fee: float) -> None:
        m = self.contract.multiplier
        self.cash -= fee + side * price * m * lots  # 买减卖增现金流（含费用）
        self.fees += fee
        if side > 0:
            self.n_buys += 1
            if self.q >= 0:
                new_q = self.q + lots
                self.avg_cost = (self.avg_cost * self.q + price * lots) / new_q
                self.q = new_q
            else:
                close = min(lots, -self.q)
                # 买入回补空头：盈利 = 开空均价 - 买回价
                self.realized_gross += close * (self.avg_cost - price) * m
                self.closed_lots += close
                self.q += lots
                if self.q > 0:
                    self.avg_cost = price
        else:
            self.n_sells += 1
            if self.q <= 0:
                new_q = self.q - lots
                self.avg_cost = (self.avg_cost * (-self.q) + price * lots) / (-new_q)
                self.q = new_q
            else:
                close = min(lots, self.q)
                self.realized_gross += close * (price - self.avg_cost) * m
                self.closed_lots += close
                self.q -= lots
                if self.q < 0:
                    self.avg_cost = price

    def equity(self, mid: float) -> float:
        return self.cash + self.q * self.contract.multiplier * mid


class Backtester:
    """回测器：多日循环驱动市场仿真 + 策略 + 风控。"""

    def __init__(
        self,
        contract: ContractSpec,
        market_params: MarketParams,
        strategy_params: StrategyParams,
        risk_params: RiskParams,
        backtest_params: BacktestParams,
        mode: str = MODE_AS,
    ):
        self.contract = contract
        self.mp = market_params
        self.sp = strategy_params
        self.rp = risk_params
        self.bp = backtest_params
        self.mode = mode
        self.schedule = SessionSchedule()

    def run(self) -> BacktestResult:
        res = BacktestResult(
            contract=self.contract,
            mode=self.mode,
            market_params=self.mp,
            strategy_params=self.sp,
            risk_params=self.rp,
            n_days=self.bp.days,
        )
        tick = self.contract.tick_size
        dt = self.mp.dt
        n_steps = int(round(TRADING_SECONDS_PER_DAY / dt))
        requote_every = max(1, self.sp.requote_every)
        sample_every = max(1, self.bp.sample_every)
        cum_pnl = 0.0
        prev_close = self.mp.price0

        for day in range(self.bp.days):
            # 每日随机流仅由 [seed, day] 决定：跨模式市场路径一致
            rng = np.random.default_rng([self.bp.seed, day])
            sim = MarketSimulator(self.contract, self.mp, rng, price0=prev_close)
            strat = ASStrategy(self.contract, self.sp, mode=self.mode)
            strat.new_day(daily_sigma=self.mp.daily_sigma)
            risk = RiskManager(self.contract, self.rp, self.schedule)
            risk.new_day(sim.prev_settlement)
            pos = PositionTracker(self.contract)

            our_bid: float | None = None
            our_ask: float | None = None
            qa_bid: float | None = None
            qa_ask: float | None = None
            rel_b: str | None = None
            rel_a: str | None = None
            ep_bid: QuoteEpisode | None = None
            ep_ask: QuoteEpisode | None = None
            day_halted = False
            mid_full: list[float] = []

            def close_ep(side: str, filled: bool) -> None:
                nonlocal ep_bid, ep_ask
                ep = ep_bid if side == "bid" else ep_ask
                if ep is not None:
                    ep.filled = filled
                    res.quote_episodes.append(ep)
                    if side == "bid":
                        ep_bid = None
                    else:
                        ep_ask = None

            def force_flatten(ts: float, step: int, note: str) -> None:
                q = pos.q
                if q == 0:
                    return
                price = sim.best_bid if q > 0 else sim.best_ask
                side = -1 if q > 0 else 1
                fee = self.contract.fee(price, abs(q))
                pos.apply(side, price, abs(q), fee)
                res.fills.append(Fill(day, ts, step, side, price, abs(q),
                                      fee, sim.mid, note=note))

            for i in range(n_steps):
                ts = (i + 1) * dt
                sim.evolve(ts)
                mid_full.append(sim.mid)

                # ---- 1) 我方挂单成交判定 ----
                if our_bid is not None or our_ask is not None:
                    fills, fill_rels, (qa_bid, qa_ask), (rel_b, rel_a) = \
                        sim.resolve_fills(our_bid, qa_bid, rel_b,
                                          our_ask, qa_ask, rel_a)
                    for (side, price), frel in zip(fills, fill_rels):
                        fee = self.contract.fee(price)
                        pos.apply(side, price, 1.0, fee)
                        res.fills.append(Fill(day, ts, i, side, price, 1.0,
                                              fee, sim.mid, rel=frel))
                        if side > 0:
                            close_ep("bid", True)
                        else:
                            close_ep("ask", True)
                else:
                    rel_b = rel_a = None

                phase = self.schedule.phase(
                    ts, self.rp.flatten_offset, self.rp.force_flatten_offset
                )

                # ---- 2) 亏损熔断检查 ----
                if not day_halted and phase == PHASE_TRADING and \
                        risk.check_loss(pos.equity(sim.mid)):
                    day_halted = True
                    res.halts.append((day, risk.state.halt_reason))
                    force_flatten(ts, i, "halt")
                    our_bid = our_ask = None
                    close_ep("bid", False)
                    close_ep("ask", False)

                # ---- 3) 报价决策 ----
                if not day_halted and phase == PHASE_TRADING and \
                        i % requote_every == 0:
                    bid, ask = strat.on_market(ts, sim.mid, pos.q)
                    bid, ask = risk.filter_quotes(bid, ask, pos.q)
                    if bid is not None and ask is not None and \
                            ask - bid < tick - 1e-9:
                        ask = bid + tick

                    if bid != our_bid:
                        close_ep("bid", False)
                        qa_bid = sim.draw_queue() if bid is not None else None
                        rel_b = None
                        if bid is not None:
                            ep_bid = QuoteEpisode(day, "bid", sim.mid - bid,
                                                  0.0, False)
                    elif ep_bid is not None:
                        ep_bid.exposure += dt

                    if ask != our_ask:
                        close_ep("ask", False)
                        qa_ask = sim.draw_queue() if ask is not None else None
                        rel_a = None
                        if ask is not None:
                            ep_ask = QuoteEpisode(day, "ask", ask - sim.mid,
                                                  0.0, False)
                    elif ep_ask is not None:
                        ep_ask.exposure += dt

                    our_bid, our_ask = bid, ask

                elif not day_halted and phase == PHASE_FLATTEN:
                    # ---- 尾盘清仓：仅报减仓方向，贴最优价 ----
                    if pos.q > 0:
                        ask = sim.best_ask - tick \
                            if sim.market_spread_ticks >= 2 else sim.best_ask
                        if ask != our_ask:
                            close_ep("ask", False)
                            qa_ask = sim.draw_queue()
                            rel_a = None
                        our_ask = ask
                        our_bid = None
                        if rel_b is not None:
                            rel_b = None
                        close_ep("bid", False)
                    elif pos.q < 0:
                        bid = sim.best_bid + tick \
                            if sim.market_spread_ticks >= 2 else sim.best_bid
                        if bid != our_bid:
                            close_ep("bid", False)
                            qa_bid = sim.draw_queue()
                            rel_b = None
                        our_bid = bid
                        our_ask = None
                        if rel_a is not None:
                            rel_a = None
                        close_ep("ask", False)
                    else:
                        our_bid = our_ask = None
                        close_ep("bid", False)
                        close_ep("ask", False)

                else:
                    # 收盘前静默阶段
                    if our_bid is not None or our_ask is not None:
                        our_bid = our_ask = None
                        close_ep("bid", False)
                        close_ep("ask", False)

                # ---- 4) 采样 ----
                if (i + 1) % sample_every == 0:
                    res.sample_ts.append(day + ts / TRADING_SECONDS_PER_DAY)
                    res.sample_equity.append(cum_pnl + pos.equity(sim.mid))
                    res.sample_inventory.append(int(pos.q))
                    res.sample_mid.append(sim.mid)
                    res.sample_bid.append(our_bid if our_bid is not None else np.nan)
                    res.sample_ask.append(our_ask if our_ask is not None else np.nan)
                    res.sample_bb.append(sim.best_bid)
                    res.sample_ba.append(sim.best_ask)

                # ---- 5) 收盘强平 ----
                if i == n_steps - 1 and pos.q != 0:
                    force_flatten(ts, i, "force")

            close_ep("bid", False)
            close_ep("ask", False)
            res.mid_full_days.append(mid_full)
            day_pnl = pos.equity(sim.mid)
            res.daily_pnl.append(day_pnl)
            cum_pnl += day_pnl
            prev_close = sim.mid

        return res
