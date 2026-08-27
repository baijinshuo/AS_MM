"""成交渠道诊断：各渠道的成交数、价格相对中间价偏移、markout。"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from as_mm import BacktestParams, Backtester, MarketParams, RiskParams, StrategyParams, get_contract

T = get_contract("T")
mp = MarketParams()
sp = StrategyParams(gamma=0.03, k_intensity=300.0)
rp = RiskParams(max_inventory=20)
bp = BacktestParams(days=2, seed=7)
res = Backtester(T, mp, sp, rp, bp, mode="as").run()

tick = T.tick_size
by_rel = defaultdict(list)
for f in res.fills:
    by_rel[f.rel].append(f)

print(f"总成交 {len(res.fills)}，总PnL {sum(res.daily_pnl):,.0f} 元")
print(f"{'渠道':<14}{'笔数':>8}{'成交价-mid(tick)':>16}{'markout1s(tick)':>16}{'markout30s(tick)':>17}")
for rel, fills in sorted(by_rel.items(), key=lambda kv: -len(kv[1])):
    offset = [f.side * (f.mid - f.price) / tick for f in fills]  # 正=有利方向
    mo1, mo30 = [], []
    for f in fills:
        arr = res.mid_full_days[f.day]
        if f.step + 2 < len(arr):
            mo1.append(f.side * (arr[f.step + 2] - f.price) / tick)
        if f.step + 60 < len(arr):
            mo30.append(f.side * (arr[f.step + 60] - f.price) / tick)
    print(f"{rel:<14}{len(fills):>8}{np.mean(offset):>16.2f}"
          f"{np.mean(mo1):>16.2f}{np.mean(mo30):>17.2f}")

# 报价与市场簿的相对位置统计（采样）
bid_pos, ask_pos = Counter(), Counter()
for i in range(len(res.sample_bid)):
    b, a = res.sample_bid[i], res.sample_ask[i]
    bb, ba = res.sample_bb[i], res.sample_ba[i]
    if not np.isnan(b):
        bid_pos[int(round((b - bb) / tick))] += 1
    if not np.isnan(a):
        ask_pos[int(round((ba - a) / tick))] += 1
print("\n我方买价相对市场买一(tick):", dict(sorted(bid_pos.items())))
print("我方卖价相对市场卖一(tick):", dict(sorted(ask_pos.items())))
