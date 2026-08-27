"""AS-MM：基于 Avellaneda-Stoikov 模型的境内国债期货做市策略。"""
from .contracts import CFEFX_TREASURY_FUTURES, ContractSpec, get_contract
from .config import (
    BacktestParams,
    MarketParams,
    RiskParams,
    StrategyParams,
    TRADING_SECONDS_PER_DAY,
)
from .model import (
    EwmaVolEstimator,
    Quotes,
    choose_gamma,
    compute_quotes,
    optimal_total_spread,
    reservation_price,
)
from .strategy import ASStrategy, MODE_AS, MODE_NAIVE, MODE_NO_SKEW
from .backtest import Backtester, BacktestResult, Fill, PositionTracker
from .analytics import compute_metrics, daily_table
from .calibration import estimate_intensity, gamma_from_inventory_budget

__version__ = "0.1.0"

__all__ = [
    "CFEFX_TREASURY_FUTURES",
    "ContractSpec",
    "get_contract",
    "BacktestParams",
    "MarketParams",
    "RiskParams",
    "StrategyParams",
    "TRADING_SECONDS_PER_DAY",
    "EwmaVolEstimator",
    "Quotes",
    "choose_gamma",
    "compute_quotes",
    "optimal_total_spread",
    "reservation_price",
    "ASStrategy",
    "MODE_AS",
    "MODE_NAIVE",
    "MODE_NO_SKEW",
    "Backtester",
    "BacktestResult",
    "Fill",
    "PositionTracker",
    "compute_metrics",
    "daily_table",
    "estimate_intensity",
    "gamma_from_inventory_budget",
    "__version__",
]
