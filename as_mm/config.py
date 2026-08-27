"""回测全局参数配置。

参数分四组：市场仿真（MarketParams）、策略（StrategyParams）、
风控（RiskParams）、回测（BacktestParams）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


# 每日交易时长（秒）：9:15-11:30 (8100s) + 13:00-15:15 (8100s)
TRADING_SECONDS_PER_DAY = 16_200.0


@dataclass
class MarketParams:
    """市场仿真参数（价格单位均为合约报价单位：元）。

    默认值按 T 主力合约量级校准：
    - 主动订单流 ~8.7 万手/日，综合日波动率 ~0.45%
    - flow_impact 为逆向选择强度（订单流冲击系数），是做市盈利环境的
      核心旋钮：冲击越强，成交的逆向选择越重，做市毛利被侵蚀越多
    """

    price0: float = 100.0            # 初始中间价
    daily_sigma: float = 0.30        # 扩散项日度波动率（价格单位/日）
    u_shape: float = 0.60            # 日内 U 型波动强度（E[phi^2]=1 归一化）
    flow_impact: float = 0.0012      # 订单流冲击系数（元/净手，逆向选择强度）
    aggressor_rate: float = 1.5      # 主动成交订单强度（单/秒/每边）
    size_p: float = 0.45             # 订单量几何分布参数（均值 1/p 手）
    spread_p: float = 0.65           # 市场价差几何分布参数（均值 1/p tick）
    spread_redraw_prob: float = 0.40 # 簿档位迁移时重抽价差的概率
    queue_mean: float = 40.0         # 触价档排队量均值（手）
    queue_floor: float = 2.0         # 触价档排队量下限（手，避免近零排队）
    flow_capture_prob: float = 0.03  # 价差内部改善报价的单笔订单捕获概率（竞争强度）
    dt: float = 0.5                  # 仿真步长（秒）
    overnight_gap_sigma: float = 0.05  # 隔夜跳空标准差（价格单位）


@dataclass
class StrategyParams:
    """AS 做市策略参数。

    gamma 为价格单位空间下的有效风险厌恶系数 Gamma = gamma_yuan * multiplier，
    见设计文档 2.3 节单位换算。
    """

    gamma: float = 0.01              # 有效风险厌恶系数 Gamma（1/价格/手）
    k_intensity: float = 100.0       # 成交强度衰减系数 k（1/价格）
    vol_sample_interval: float = 5.0 # 波动率采样间隔（秒）
    vol_halflife: float = 360.0      # EWMA 半衰期（秒）
    requote_every: int = 1           # 每隔多少步重新报价
    max_spread_ticks: int = 12       # 报价总价差上限（tick），防波动率尖峰
    naive_half_ticks: int = 1        # 朴素基准：固定半价差（tick）


@dataclass
class RiskParams:
    """风控参数。"""

    max_inventory: int = 20          # 最大净库存（手），触界单边报价
    daily_loss_limit: float = 200_000.0  # 单日亏损限额（元），触限停止报价并平仓
    flatten_offset: float = 900.0    # 收盘前 N 秒进入清仓模式
    force_flatten_offset: float = 30.0   # 收盘前 N 秒停止报价，仅留强平


@dataclass
class BacktestParams:
    """回测控制参数。"""

    days: int = 10
    seed: int = 7
    sample_every: int = 20           # 采样间隔（步），20 步 = 10 秒


DEFAULT_MARKET = MarketParams()
DEFAULT_STRATEGY = StrategyParams()
DEFAULT_RISK = RiskParams()
DEFAULT_BACKTEST = BacktestParams()


def preset_for(symbol: str) -> dict:
    """按品种给出建议参数预设（gamma/k 由参数扫描校准，见 README）。"""
    base = {
        "TS": {"daily_sigma": 0.08, "gamma": 0.008, "k": 100.0},
        "TF": {"daily_sigma": 0.18, "gamma": 0.010, "k": 100.0},
        "T": {"daily_sigma": 0.30, "gamma": 0.010, "k": 100.0},
        "TL": {"daily_sigma": 0.55, "gamma": 0.012, "k": 100.0},
    }[symbol.upper()]
    return {
        "market": replace(DEFAULT_MARKET, daily_sigma=base["daily_sigma"]),
        "strategy": replace(DEFAULT_STRATEGY, gamma=base["gamma"],
                            k_intensity=base["k"]),
    }
