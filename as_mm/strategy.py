"""AS 做市策略：每步生成双边报价。

三种模式用于消融对比：
- 'as'      完整 Avellaneda-Stoikov：库存偏移（保留价）+ 最优价差
- 'no_skew' 仅最优价差，报价中枢固定在中间价（无库存管理）
- 'naive'   朴素固定 tick 报价，无任何模型成分
"""
from __future__ import annotations

from .config import StrategyParams, TRADING_SECONDS_PER_DAY
from .contracts import ContractSpec
from .model import (
    EwmaVolEstimator,
    compute_quotes,
    implied_seed_sigma,
    floor_to_tick,
    ceil_to_tick,
)

MODE_AS = "as"
MODE_NO_SKEW = "no_skew"
MODE_NAIVE = "naive"


class ASStrategy:
    """Avellaneda-Stoikov 做市策略。"""

    def __init__(
        self,
        contract: ContractSpec,
        params: StrategyParams,
        mode: str = MODE_AS,
        daily_sigma: float | None = None,
    ):
        if mode not in (MODE_AS, MODE_NO_SKEW, MODE_NAIVE):
            raise ValueError(f"未知策略模式 {mode!r}")
        self.contract = contract
        self.p = params
        self.mode = mode
        self.daily_sigma = daily_sigma
        self.vol = EwmaVolEstimator(
            sample_interval=params.vol_sample_interval,
            halflife=params.vol_halflife,
        )

    def new_day(self, daily_sigma: float | None = None) -> None:
        """重置波动率估计器（以上一日或配置的日波动率为种子）。"""
        if daily_sigma is not None:
            self.daily_sigma = daily_sigma
        seed = implied_seed_sigma(self.daily_sigma) if self.daily_sigma else None
        self.vol.reset(seed_sigma=seed)

    def on_market(self, ts: float, mid: float, q: int) -> tuple[float | None, float | None]:
        """输入当前交易秒、中间价与库存，输出 (bid, ask)。"""
        sigma = self.vol.update(ts, mid)
        if sigma <= 0:
            return None, None

        tick = self.contract.tick_size
        tau = max(TRADING_SECONDS_PER_DAY - ts, 0.0)

        if self.mode == MODE_NAIVE:
            half = self.p.naive_half_ticks * tick
            bid = floor_to_tick(mid - half, tick)
            ask = ceil_to_tick(mid + half, tick)
            if ask - bid < tick - 1e-9:
                ask = bid + tick
            return bid, ask

        if self.mode == MODE_NO_SKEW:
            q_eff = 0  # 报价中枢不随库存偏移，价差仍按 AS 公式
        else:
            q_eff = q

        quotes = compute_quotes(
            mid=mid,
            q=q_eff,
            tau=tau,
            sigma=sigma,
            gamma=self.p.gamma,
            k=self.p.k_intensity,
            tick=tick,
            max_spread_ticks=self.p.max_spread_ticks,
        )
        return quotes.bid, quotes.ask
