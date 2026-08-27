"""风控模块：时段管理、库存界限、价格带、止损、尾盘清仓。

交易时段：9:15-11:30、13:00-15:15（中金所国债期货）。
模型内部使用"交易秒"连续计时：两个时段首尾相接共 16200 秒，
AS 公式中的剩余时间 tau 即剩余交易秒。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RiskParams, TRADING_SECONDS_PER_DAY
from .contracts import ContractSpec

# 墙钟时段（秒，自当日 0 点起），仅用于展示换算
WALL_CLOCK_SESSIONS = [
    (9 * 3600 + 15 * 60, 11 * 3600 + 30 * 60),
    (13 * 3600, 15 * 3600 + 15 * 60),
]

PHASE_TRADING = "trading"   # 正常做市
PHASE_FLATTEN = "flatten"   # 尾盘清仓
PHASE_CLOSED = "closed"     # 停止报价（收盘前最后时刻）


class SessionSchedule:
    """交易时段调度。"""

    total_seconds = TRADING_SECONDS_PER_DAY

    def phase(self, ts: float, flatten_offset: float, force_offset: float) -> str:
        if ts >= self.total_seconds - force_offset:
            return PHASE_CLOSED
        if ts >= self.total_seconds - flatten_offset:
            return PHASE_FLATTEN
        return PHASE_TRADING

    @staticmethod
    def wall_clock(ts: float) -> str:
        """交易秒 -> 墙钟时间字符串（跨午休连续映射，仅供展示）。"""
        for start, end in WALL_CLOCK_SESSIONS:
            length = end - start
            if ts < length:
                t = start + ts
                break
            ts -= length
        else:
            t = WALL_CLOCK_SESSIONS[-1][1]
        return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


@dataclass
class RiskState:
    """单日风控状态。"""

    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    """报价过滤器：库存界限 -> 涨跌停价格带 -> 单边撤单。"""

    def __init__(self, contract: ContractSpec, params: RiskParams,
                 schedule: SessionSchedule | None = None):
        self.contract = contract
        self.p = params
        self.schedule = schedule or SessionSchedule()
        self.state = RiskState()
        self._band_low = 0.0
        self._band_high = float("inf")

    def new_day(self, prev_settlement: float) -> None:
        """按上一结算价重置日内风控与价格带。"""
        self.state = RiskState()
        lim = self.contract.price_limit_pct
        self._band_low = prev_settlement * (1.0 - lim)
        self._band_high = prev_settlement * (1.0 + lim)

    def check_loss(self, day_pnl: float) -> bool:
        """单日亏损检查（返回是否触发熔断）。"""
        if not self.state.halted and day_pnl <= -self.p.daily_loss_limit:
            self.state.halted = True
            self.state.halt_reason = (
                f"单日亏损 {day_pnl:,.0f} 元触限 {self.p.daily_loss_limit:,.0f} 元"
            )
        return self.state.halted

    def filter_quotes(
        self, bid: float | None, ask: float | None, q: int
    ) -> tuple[float | None, float | None]:
        """对策略报价应用风控过滤。

        1) 涨跌停价格带裁剪；
        2) 库存触界：撤掉继续加仓的一侧（|q| >= max_inventory）。
        """
        if self.state.halted:
            return None, None

        tick = self.contract.tick_size
        if bid is not None:
            if bid > self._band_high or bid < self._band_low:
                bid = min(max(bid, self._band_low), self._band_high)
                bid = round(bid / tick) * tick
        if ask is not None:
            if ask > self._band_high or ask < self._band_low:
                ask = min(max(ask, self._band_low), self._band_high)
                ask = round(ask / tick) * tick

        if q >= self.p.max_inventory:
            bid = None  # 多头库存触界：撤买报价
        elif q <= -self.p.max_inventory:
            ask = None  # 空头库存触界：撤卖报价
        return bid, ask
