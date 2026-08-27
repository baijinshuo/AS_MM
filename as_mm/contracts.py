"""中金所国债期货合约规格。

参数以交易所公布规则为基准整理，实际交易请以交易所最新公告为准。
所有参数均为可配置项，可按需覆盖。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """单张合约规格（价格单位：元；乘数单位：元/点）。"""

    symbol: str
    name: str
    notional: float          # 合约面值（元）
    tick_size: float         # 最小变动价位（元）
    multiplier: float        # 合约乘数（元/点）
    price_limit_pct: float   # 每日价格最大波动限制（比例）
    margin_rate: float       # 最低交易保证金率（比例）
    fee_rate: float          # 交易手续费率（按成交金额比例，开平同率）

    @property
    def tick_value(self) -> float:
        """1 个最小变动价位的合约价值（元）。"""
        return self.tick_size * self.multiplier

    def fee(self, price: float, lots: float = 1.0) -> float:
        """单笔成交手续费（元）。"""
        return price * self.multiplier * abs(lots) * self.fee_rate

    def notional_value(self, price: float, lots: float = 1.0) -> float:
        return price * self.multiplier * abs(lots)

    def margin(self, price: float, lots: float = 1.0) -> float:
        return self.notional_value(price, lots) * self.margin_rate


#: 中金所国债期货四品种（2/5/10/30 年期）
CFEFX_TREASURY_FUTURES: dict[str, ContractSpec] = {
    "TS": ContractSpec(
        symbol="TS", name="2年期国债期货", notional=2_000_000,
        tick_size=0.005, multiplier=20_000,
        price_limit_pct=0.005, margin_rate=0.005, fee_rate=1e-5,
    ),
    "TF": ContractSpec(
        symbol="TF", name="5年期国债期货", notional=1_000_000,
        tick_size=0.005, multiplier=10_000,
        price_limit_pct=0.012, margin_rate=0.012, fee_rate=1e-5,
    ),
    "T": ContractSpec(
        symbol="T", name="10年期国债期货", notional=1_000_000,
        tick_size=0.005, multiplier=10_000,
        price_limit_pct=0.020, margin_rate=0.020, fee_rate=1e-5,
    ),
    "TL": ContractSpec(
        symbol="TL", name="30年期国债期货", notional=1_000_000,
        tick_size=0.01, multiplier=10_000,
        price_limit_pct=0.035, margin_rate=0.035, fee_rate=1e-5,
    ),
}


def get_contract(symbol: str) -> ContractSpec:
    try:
        return CFEFX_TREASURY_FUTURES[symbol.upper()]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(
            f"未知合约 {symbol!r}，可选: {sorted(CFEFX_TREASURY_FUTURES)}"
        ) from exc
