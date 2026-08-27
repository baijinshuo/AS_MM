"""成交动力学诊断：报价关系分布、成交渠道、市场价差分布。"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from as_mm import BacktestParams, Backtester, MarketParams, RiskParams, StrategyParams, get_contract
from as_mm.config import TRADING_SECONDS_PER_DAY

T = get_contract("T")
mp = MarketParams(price0=100.0, daily_sigma=0.30)
sp = StrategyParams(gamma=0.03, k_intensity=300.0)
rp = RiskParams(max_inventory=10)
bp = BacktestParams(days=1, seed=7, sample_every=20)

res = Backtester(T, mp, sp, rp, bp, mode="as").run()
print(f"fills={len(res.fills)}  episodes={len(res.quote_episodes)} "
      f"filled_eps={sum(e.filled for e in res.quote_episodes)}")

# 市场价差与报价位置分布（从采样重建）
import math
tick = T.tick_size
n_steps = int(TRADING_SECONDS_PER_DAY / mp.dt)
rng = np.random.default_rng([7, 0])
from as_mm.market_sim import MarketSimulator

sim = MarketSimulator(T, mp, rng, price0=100.0)
spreads = Counter()
for i in range(n_steps):
    sim.evolve((i + 1) * mp.dt)
    spreads[sim.market_spread_ticks] += 1
print("市场价差分布(tick):", dict(sorted(spreads.items())))

# 中间价每步移动分布
mids = [100.0]
rng2 = np.random.default_rng([7, 0])
sim2 = MarketSimulator(T, mp, rng2, price0=100.0)
for i in range(n_steps):
    st = sim2.evolve((i + 1) * mp.dt)
    mids.append(st.mid)
mids = np.array(mids)
dm = np.diff(mids)
print(f"每步中间价移动: std={dm.std():.6f} ({dm.std()/tick:.2f} tick), "
      f"|dm|>1tick 占比 {np.mean(np.abs(dm) > tick):.3f}")

# 报价距离分析（episode delta 分布）
deltas = [e.delta / tick for e in res.quote_episodes]
d_arr = np.array(deltas)
print(f"挂单delta分布(tick): mean={d_arr.mean():.2f} "
      f"p10={np.percentile(d_arr,10):.2f} p50={np.percentile(d_arr,50):.2f} "
      f"p90={np.percentile(d_arr,90):.2f}")
print(f"成交episode的delta(tick): mean="
      f"{np.mean([e.delta/tick for e in res.quote_episodes if e.filled]):.2f}")
print(f"未成交delta(tick): mean="
      f"{np.mean([e.delta/tick for e in res.quote_episodes if not e.filled]):.2f}")
exps = [e.exposure for e in res.quote_episodes]
print(f"episode暴露时长: mean={np.mean(exps):.2f}s p90={np.percentile(exps,90):.2f}s")

# 成交笔记分布
notes = Counter(f.note for f in res.fills)
print("成交类型:", dict(notes))
sides = Counter(f.side for f in res.fills)
print("方向:", {("+1买" if s > 0 else "-1卖"): c for s, c in sides.items()})
print(f"逐日PnL: {res.daily_pnl}")
