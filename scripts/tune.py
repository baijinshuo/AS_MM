"""市场环境细调：网格搜索兼顾 (a) 真实价格波动率 (b) 合理做市利润。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from as_mm import BacktestParams, Backtester, MarketParams, RiskParams, StrategyParams, compute_metrics, get_contract
from as_mm.config import TRADING_SECONDS_PER_DAY

T = get_contract("T")

rows = []
grid = []
for mu in (1.5, 2.5):
    for eta in (0.0006, 0.0008, 0.0010):
        for spread_p in (0.45, 0.55):
            grid.append((mu, eta, spread_p))

print(f"{'mu':>5}{'eta':>8}{'spr_p':>7}{'日波动':>8}{'PnL/日':>10}{'RT/日':>8}"
      f"{'净/RT':>8}{'均|q|':>7}{'Sharpe':>8}")
for mu, eta, spread_p in grid:
    mp = MarketParams(price0=100.0, daily_sigma=0.30, queue_mean=25.0,
                      flow_impact=eta, flow_capture_prob=0.04,
                      aggressor_rate=mu, spread_p=spread_p)
    sp = StrategyParams(gamma=0.03, k_intensity=300.0)
    rp = RiskParams(max_inventory=20)
    bp = BacktestParams(days=4, seed=7)
    res = Backtester(T, mp, sp, rp, bp, mode="as").run()
    m = compute_metrics(res)
    # 实际日波动率
    dvol = np.mean([np.std(np.diff(d)) / np.sqrt(mp.dt) * np.sqrt(TRADING_SECONDS_PER_DAY)
                    for d in res.mid_full_days])
    print(f"{mu:>5.1f}{eta:>8.4f}{spread_p:>7.2f}{dvol:>8.3f}"
          f"{m['total_pnl']/4:>10,.0f}{m['closed_lots']/4:>8,.0f}"
          f"{m['net_per_roundtrip_yuan']:>8,.1f}{m['avg_abs_inventory']:>7.1f}"
          f"{m['sharpe_annual']:>8.1f}")
