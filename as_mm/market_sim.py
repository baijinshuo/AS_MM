"""市场仿真器：中间价过程 + 订单簿 + 主动订单流 + 我方限价单成交判定。

设计要点（详见设计文档第 4 节）：
- 中间价 = 扩散项 + 订单流冲击项（后者是逆向选择的来源）
- 市场最优买卖价围绕中间价在 tick 网格上移动，价差服从几何分布
- 我方报价与市场簿合并后按关系判定成交：
  * 穿越市场对方价 -> 立即按对方价成交（市价化）
  * 价差内部改善 -> 每笔对向订单以 flow_capture_prob 概率成交（竞争）
  * 恰在触价档 -> 排在既有排队量之后，队列耗尽且订单量更大时成交
  * 深于触价档 -> 不直接成交；当簿档位下移到我方价位时视为先到（排队量=0）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import MarketParams, TRADING_SECONDS_PER_DAY
from .contracts import ContractSpec


@dataclass
class EvolvedStep:
    """单步演化后的市场状态快照。"""

    mid: float
    best_bid: float
    best_ask: float
    sells: list[int]  # 本步各笔市价卖单量（手）
    buys: list[int]   # 本步各笔市价买单量（手）


# 我方报价与市场簿的关系
REL_MARKETABLE = "marketable"  # 穿越对方最优价
REL_IMPROVE = "improve"        # 价差内部改善
REL_TOUCH = "touch"            # 恰在触价档（排队）
REL_DEEP = "deep"              # 深于触价档


class MarketSimulator:
    """单日市场仿真（一个实例对应一个交易日）。"""

    def __init__(
        self,
        contract: ContractSpec,
        params: MarketParams,
        rng: np.random.Generator,
        price0: float | None = None,
    ):
        self.contract = contract
        self.p = params
        self.rng = rng
        self.tick = contract.tick_size

        p0 = params.price0 if price0 is None else price0
        gap = float(rng.normal(0.0, params.overnight_gap_sigma))
        self.mid: float = p0 + gap
        self.prev_settlement = p0

        # 初始订单簿（先设初值，预生成随机池后再执行簿档位追踪）
        m0 = max(1, int(rng.geometric(params.spread_p)))
        self._bb_ticks = int(math.floor(self.mid / self.tick - m0 / 2.0))
        self._spread_ticks = m0

        # 最近一步订单流（供成交判定使用）
        self._last_sells: list[int] = []
        self._last_buys: list[int] = []

        self.n_steps = int(round(TRADING_SECONDS_PER_DAY / params.dt))
        self._pregenerate()
        self._chase_book()

    # ------------------------------------------------------------ 随机数预生成
    def _pregenerate(self) -> None:
        n = self.n_steps
        dt = self.p.dt
        lam = self.p.aggressor_rate * dt
        self._normals = self.rng.normal(0.0, 1.0, n).tolist()
        self._n_sells = self.rng.poisson(lam, n).tolist()
        self._n_buys = self.rng.poisson(lam, n).tolist()
        total_orders = max(int(sum(self._n_sells) + sum(self._n_buys)), 1)
        # 订单量池（几何分布，>=1 手）
        self._sizes = self.rng.geometric(
            self.p.size_p, int(total_orders * 1.5) + 64
        ).tolist()
        # 排队量池 / 价差重抽池 / 档位迁移判定池
        self._queues = self.rng.exponential(self.p.queue_mean, n * 2 + 64).tolist()
        self._spread_draws = self.rng.geometric(self.p.spread_p, n + 64).tolist()
        self._shift_unis = self.rng.random(n * 8 + 64).tolist()
        # 改善成交判定：每步固定 16 个均匀数（Poisson(1.25) 超过 16 概率≈0），
        # 按步索引消耗，与策略行为解耦，保证跨模式市场路径一致。
        self._uniforms = self.rng.random((n, 16)).tolist()
        self._si = 0
        self._qi = 0
        self._di = 0
        self._hi = 0
        self._step = 0
        self._cur_step = 0

    # ------------------------------------------------------------ 订单簿
    @property
    def best_bid(self) -> float:
        return self._bb_ticks * self.tick

    @property
    def best_ask(self) -> float:
        return (self._bb_ticks + self._spread_ticks) * self.tick

    @property
    def market_spread_ticks(self) -> int:
        return self._spread_ticks

    def _chase_book(self) -> None:
        """簿档位追踪中间价：中间价逼近边界时整体迁移并可能重抽价差。"""
        tick = self.tick
        guard = 0
        while self.mid > self.best_ask - 0.25 * tick and guard < 200:
            self._shift_book(+1)
            guard += 1
        guard = 0
        while self.mid < self.best_bid + 0.25 * tick and guard < 200:
            self._shift_book(-1)
            guard += 1

    def _shift_book(self, direction: int) -> None:
        self._bb_ticks += direction
        if self._hi < len(self._shift_unis) and \
                self._shift_unis[self._hi] < self.p.spread_redraw_prob:
            if self._di < len(self._spread_draws):
                self._spread_ticks = max(1, int(self._spread_draws[self._di]))
                self._di += 1
        self._hi += 1

    # ------------------------------------------------------------ 演化
    def evolve(self, ts: float) -> EvolvedStep:
        """单步演化：生成订单流 -> 更新中间价 -> 迁移订单簿。"""
        i = self._step
        p = self.p
        dt = p.dt

        n_s = int(self._n_sells[i])
        n_b = int(self._n_buys[i])
        sells = self._take_sizes(n_s)
        buys = self._take_sizes(n_b)
        net = sum(buys) - sum(sells)

        # 日内 U 型波动率（E[phi^2] = 1 归一化）
        x = ts / TRADING_SECONDS_PER_DAY
        phi2 = 1.0 + p.u_shape * (4.0 * (x - 0.5) ** 2 - 1.0 / 3.0)
        phi2 = max(phi2, 0.1)
        sigma_t = p.daily_sigma / math.sqrt(TRADING_SECONDS_PER_DAY) * math.sqrt(phi2)

        self.mid += p.flow_impact * net + sigma_t * math.sqrt(dt) * self._normals[i]
        self._chase_book()
        self._last_sells = sells
        self._last_buys = buys
        self._cur_step = i
        self._step += 1
        return EvolvedStep(self.mid, self.best_bid, self.best_ask, sells, buys)

    def _take_sizes(self, n: int) -> list[int]:
        if n <= 0:
            return []
        out = []
        for _ in range(n):
            out.append(int(self._sizes[self._si]) if self._si < len(self._sizes) else 1)
            if self._si < len(self._sizes) - 1:
                self._si += 1
        return out

    def draw_queue(self) -> float:
        """新挂单在触价档身前的排队量（手），带下限避免近零排队。"""
        q = self._queues[self._qi] if self._qi < len(self._queues) else self.p.queue_mean
        if self._qi < len(self._queues) - 1:
            self._qi += 1
        return max(float(q), getattr(self.p, "queue_floor", 0.0))

    # ------------------------------------------------------------ 成交判定
    def resolve_fills(
        self,
        bid: float | None,
        qa_bid: float | None,
        prev_rel_bid: str | None,
        ask: float | None,
        qa_ask: float | None,
        prev_rel_ask: str | None,
    ) -> tuple[list[tuple[int, float]], list[str],
               tuple[float | None, float | None], tuple[str | None, str | None]]:
        """对我方两个挂单执行成交判定。

        返回 (fills, (qa_bid, qa_ask), (rel_bid, rel_ask))；
        fills 中每项为 (side, price, relation)，side=+1 买入 / -1 卖出，
        relation 为成交渠道（marketable/improve/touch）。
        """
        fills: list[tuple[int, float]] = []
        rels_fill: list[str] = []
        rel_b: str | None = None
        rel_a: str | None = None

        if bid is not None:
            filled, fpx, qa_bid, rel_b = self._resolve_side(
                bid, qa_bid, prev_rel_bid, is_bid=True
            )
            if filled:
                fills.append((+1, fpx))
                rels_fill.append(rel_b)
        if ask is not None:
            filled, fpx, qa_ask, rel_a = self._resolve_side(
                ask, qa_ask, prev_rel_ask, is_bid=False
            )
            if filled:
                fills.append((-1, fpx))
                rels_fill.append(rel_a)

        return fills, rels_fill, (qa_bid, qa_ask), (rel_b, rel_a)

    def _resolve_side(
        self,
        price: float,
        q_ahead: float | None,
        prev_rel: str | None,
        is_bid: bool,
    ) -> tuple[bool, float, float | None, str]:
        """判定单边挂单。

        返回 (filled, fill_price, q_ahead, relation)；
        未成交时 fill_price 为原挂价、q_ahead 为更新后的排队量。
        """
        bb, ba = self.best_bid, self.best_ask
        t = self.tick
        flow = self._last_sells if is_bid else self._last_buys

        if is_bid:
            d_touch = int(round((price - bb) / t))  # 相对买方触价档的 tick 数
            crossed = price >= ba - 1e-9
        else:
            d_touch = int(round((ba - price) / t))  # 相对卖方触价档的 tick 数
            crossed = price <= bb + 1e-9

        # 1) 穿越对方最优价：立即按对方挂价成交（撮合取先挂方价格）
        if crossed:
            return True, (ba if is_bid else bb), None, REL_MARKETABLE

        if d_touch > 0:
            # 2) 价差内部改善：每笔对向订单按捕获概率成交
            pi = self.p.flow_capture_prob
            row = self._uniforms[self._cur_step] \
                if self._cur_step < len(self._uniforms) else [0.5] * 16
            for j in range(len(flow)):
                u = row[j] if j < len(row) else 0.5
                if u < pi:
                    return True, price, q_ahead, REL_IMPROVE
            return False, price, q_ahead, REL_IMPROVE

        if d_touch == 0:
            # 3) 触价档排队：簿档位迁移到我方价位时视为先到（排队量清零）
            if prev_rel == REL_DEEP:
                q_ahead = 0.0
            ahead = q_ahead if q_ahead is not None else 0.0
            for size in flow:
                if size > ahead:
                    return True, price, None, REL_TOUCH
                ahead -= size
            return False, price, ahead, REL_TOUCH

        # 4) 深于触价档：本步不成交
        return False, price, q_ahead, REL_DEEP
