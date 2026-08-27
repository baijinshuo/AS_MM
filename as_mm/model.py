"""Avellaneda-Stoikov (2008) 做市模型核心。

理论结果（近似解，工程标准实现）：
    保留价格   r = P - q * Gamma * sigma^2 * tau
    最优总价差 delta = Gamma * sigma^2 * tau + (2/Gamma) * ln(1 + Gamma/k)
    双边报价   bid = r - delta/2,  ask = r + delta/2

其中 P 为中间价，q 为库存（手，多头为正），tau 为剩余交易时间（秒），
sigma 为每 sqrt(秒) 的价格波动率，Gamma = gamma_yuan * multiplier 为价格
单位空间下的有效风险厌恶系数，k 为成交强度衰减系数（1/价格单位），
成交强度 lambda(delta) = A * exp(-k * delta)。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import TRADING_SECONDS_PER_DAY

_EPS = 1e-9


# ---------------------------------------------------------------- 工具函数
def floor_to_tick(x: float, tick: float) -> float:
    """向下取整到 tick 网格（带浮点容差）。"""
    return math.floor(x / tick + _EPS) * tick


def ceil_to_tick(x: float, tick: float) -> float:
    """向上取整到 tick 网格（带浮点容差）。"""
    return math.ceil(x / tick - _EPS) * tick


def ticks_between(a: float, b: float, tick: float) -> int:
    """(a - b) / tick 四舍五入为整数（同网格价格比较）。"""
    return int(round((a - b) / tick))


# ---------------------------------------------------------------- AS 公式
def inventory_skew_per_lot(tau: float, sigma: float, gamma: float) -> float:
    """每手库存的保留价格偏移量（价格单位）：Gamma * sigma^2 * tau。"""
    return gamma * sigma * sigma * tau


def reservation_price(
    mid: float, q: float, tau: float, sigma: float, gamma: float
) -> float:
    """保留价格：r = P - q * Gamma * sigma^2 * tau。

    库存越多（q>0），报价中枢越低，以鼓励卖出、回收库存。
    """
    return mid - q * inventory_skew_per_lot(tau, sigma, gamma)


def optimal_total_spread(tau: float, sigma: float, gamma: float, k: float) -> float:
    """最优总报价价差（bid/ask 之间的距离，价格单位）。

    delta = Gamma * sigma^2 * tau + (2/Gamma) * ln(1 + Gamma/k)
    """
    return gamma * sigma * sigma * tau + (2.0 / gamma) * math.log1p(gamma / k)


@dataclass(frozen=True)
class Quotes:
    """单步双边报价（价格单位，已取整到 tick 网格）。"""

    bid: float
    ask: float
    reservation: float
    total_spread: float
    skew: float  # q * Gamma * sigma^2 * tau，库存导致的整体偏移


def compute_quotes(
    mid: float,
    q: float,
    tau: float,
    sigma: float,
    gamma: float,
    k: float,
    tick: float,
    max_spread_ticks: int = 12,
) -> Quotes:
    """生成 AS 双边报价：保留价 ± 最优半价差，取整到 tick 网格。"""
    skew = q * inventory_skew_per_lot(tau, sigma, gamma)
    r = mid - skew
    total = optimal_total_spread(tau, sigma, gamma, k)
    total = min(total, max_spread_ticks * tick)

    bid = floor_to_tick(r - total / 2.0, tick)
    ask = ceil_to_tick(r + total / 2.0, tick)
    if ask - bid < tick - _EPS:
        # 取整后价差过窄：以保留价为中心保证至少 1 tick
        bid = floor_to_tick(r, tick)
        ask = bid + tick
    return Quotes(bid=bid, ask=ask, reservation=r, total_spread=ask - bid, skew=skew)


def choose_gamma(
    sigma: float,
    tau: float,
    q_max: int,
    tick: float,
    skew_ticks: float = 1.0,
) -> float:
    """由库存风险预算反解有效风险厌恶系数 Gamma。

    目标：库存打满 q_max 时保留价偏移 skew_ticks 个 tick。
    Gamma = skew_ticks * tick / (sigma^2 * tau * q_max)
    """
    if q_max <= 0 or tau <= 0 or sigma <= 0:
        raise ValueError("q_max/tau/sigma 必须为正")
    return skew_ticks * tick / (sigma * sigma * tau * q_max)


def implied_seed_sigma(daily_sigma: float) -> float:
    """日度波动率换算为每 sqrt(秒) 波动率（用于 EWMA 种子）。"""
    return daily_sigma / math.sqrt(TRADING_SECONDS_PER_DAY)


# ---------------------------------------------------------------- 波动率估计
class EwmaVolEstimator:
    """基于固定间隔中间价收益的 EWMA 已实现波动率估计器。

    输出为每 sqrt(秒) 的价格波动率（与 AS 公式中 tau 的秒单位匹配）。
    """

    def __init__(
        self,
        sample_interval: float = 5.0,
        halflife: float = 360.0,
        seed_sigma: float | None = None,
        floor_mult: float = 0.3,
        cap_mult: float = 4.0,
    ):
        self.sample_interval = sample_interval
        self.lam = 0.5 ** (sample_interval / halflife)  # 每样本衰减因子
        self.seed_sigma = seed_sigma
        self.floor_mult = floor_mult
        self.cap_mult = cap_mult
        self._var: float | None = None
        self._last_price: float | None = None
        self._last_t: float | None = None

    def reset(self, seed_sigma: float | None = None) -> None:
        if seed_sigma is not None:
            self.seed_sigma = seed_sigma
        self._var = self.seed_sigma**2 if self.seed_sigma else None
        self._last_price = None
        self._last_t = None

    def update(self, t: float, price: float) -> float:
        """输入当前交易秒与中间价，返回当前每 sqrt(秒) 波动率估计。"""
        if self._last_price is None:
            self._last_price = price
            self._last_t = t
        elif t - self._last_t >= self.sample_interval - 1e-9:
            dt = t - self._last_t
            ret = price - self._last_price
            var_per_sec = ret * ret / dt
            if self._var is None:
                self._var = var_per_sec
            else:
                self._var = self.lam * self._var + (1.0 - self.lam) * var_per_sec
            self._last_price = price
            self._last_t = t

        if self._var is None:
            return self.seed_sigma if self.seed_sigma else 0.0
        sigma = math.sqrt(self._var)
        if self.seed_sigma:
            sigma = min(max(sigma, self.floor_mult * self.seed_sigma),
                        self.cap_mult * self.seed_sigma)
        return sigma
