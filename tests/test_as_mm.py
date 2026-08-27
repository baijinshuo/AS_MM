"""AS-MM 单元测试：模型数学、记账、回测不变量、校准。

可直接 `python -m pytest tests/ -q` 或 `python tests/test_as_mm.py` 运行。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from as_mm import (
    BacktestParams,
    Backtester,
    MarketParams,
    RiskParams,
    StrategyParams,
    compute_metrics,
    get_contract,
)
from as_mm.backtest import Fill, PositionTracker, QuoteEpisode
from as_mm.calibration import estimate_intensity, gamma_from_inventory_budget
from as_mm.config import TRADING_SECONDS_PER_DAY
from as_mm.model import (
    EwmaVolEstimator,
    choose_gamma,
    compute_quotes,
    optimal_total_spread,
    reservation_price,
)
from as_mm.risk import PHASE_CLOSED, PHASE_FLATTEN, PHASE_TRADING, RiskManager, SessionSchedule
from as_mm.strategy import MODE_AS, MODE_NAIVE, MODE_NO_SKEW

T = get_contract("T")


# ---------------------------------------------------------------- 模型数学
def test_reservation_price():
    s, sigma, tau, g = 100.0, 0.002, 8100.0, 0.03
    skew = g * sigma**2 * tau
    r_pos = reservation_price(s, 5, tau, sigma, g)
    r_neg = reservation_price(s, -5, tau, sigma, g)
    assert r_pos == s - 5 * skew
    assert r_neg == s + 5 * skew
    assert abs(r_pos + r_neg - 2 * s) < 1e-12  # 关于零库存对称
    assert r_pos < r_neg  # 多头库存压低报价中枢


def test_optimal_spread():
    sigma, tau, g, k = 0.002, 8100.0, 0.03, 300.0
    expected = g * sigma**2 * tau + (2 / g) * math.log1p(g / k)
    assert abs(optimal_total_spread(tau, sigma, g, k) - expected) < 1e-12
    # tau -> 0 退化为纯强度项
    assert abs(optimal_total_spread(0, sigma, g, k) - (2 / g) * math.log1p(g / k)) < 1e-12
    # 对 sigma / tau 单调递增
    assert optimal_total_spread(tau, sigma * 2, g, k) > optimal_total_spread(tau, sigma, g, k)
    assert optimal_total_spread(tau + 1, sigma, g, k) > optimal_total_spread(tau, sigma, g, k)


def test_compute_quotes_grid_and_skew():
    tick = T.tick_size
    q0 = compute_quotes(100.123456, 0, 8100.0, 0.003, 0.05, 300.0, tick)
    q5 = compute_quotes(100.123456, 5, 8100.0, 0.003, 0.05, 300.0, tick)
    for q in (q0, q5):
        assert abs(q.bid / tick - round(q.bid / tick)) < 1e-6
        assert abs(q.ask / tick - round(q.ask / tick)) < 1e-6
        assert q.ask - q.bid >= tick - 1e-9
    # 多头库存整体下移报价
    assert q5.bid <= q0.bid
    assert q5.ask <= q0.ask
    assert q5.reservation < q0.reservation


def test_choose_gamma_roundtrip():
    sigma, tau, qmax, skew_ticks = 0.002, 8100.0, 20, 1.0
    g = choose_gamma(sigma, tau, qmax, T.tick_size, skew_ticks)
    assert g > 0
    # 打满库存时偏移恰为 skew_ticks 个 tick
    skew = qmax * g * sigma**2 * tau
    assert abs(skew / T.tick_size - skew_ticks) < 1e-9
    g2 = gamma_from_inventory_budget(sigma, tau, qmax, T.tick_size, skew_ticks)
    assert abs(g2 - g) < 1e-15


def test_ewma_vol_estimator():
    est = EwmaVolEstimator(sample_interval=5.0, halflife=360.0, seed_sigma=0.002)
    est.reset(seed_sigma=0.002)
    # 前几个样本返回种子
    assert est.update(0.0, 100.0) == 0.002
    # 常数价格 -> 波动率衰减
    v = est.update(5.0, 100.0)
    assert v < 0.002
    # 大幅波动 -> 波动率上升但有上限
    est2 = EwmaVolEstimator(5.0, 360.0, seed_sigma=0.002)
    est2.reset(seed_sigma=0.002)
    est2.update(0.0, 100.0)
    v2 = est2.update(5.0, 100.5)  # 单样本 5s 移动 0.5
    assert v2 == 0.008  # 4x cap


# ---------------------------------------------------------------- 记账
def test_position_tracker_roundtrip():
    pos = PositionTracker(T)
    fee = T.fee(100.0)
    pos.apply(+1, 100.0, 1.0, fee)   # 买 1 手 @100.000
    pos.apply(-1, 100.005, 1.0, fee)  # 卖 1 手 @100.005
    assert pos.q == 0
    assert abs(pos.realized_gross - 0.005 * T.multiplier) < 1e-9
    assert abs(pos.equity(100.0) - (0.005 * T.multiplier - 2 * fee)) < 1e-9
    assert pos.closed_lots == 1.0

    # 空头回合
    pos2 = PositionTracker(T)
    pos2.apply(-1, 100.010, 1.0, fee)
    pos2.apply(+1, 100.000, 1.0, fee)
    assert abs(pos2.realized_gross - 0.010 * T.multiplier) < 1e-9
    assert pos2.q == 0


def test_position_tracker_partial():
    pos = PositionTracker(T)
    fee = T.fee(100.0)
    pos.apply(+1, 100.0, 2.0, fee)
    pos.apply(-1, 100.005, 1.0, fee)
    assert pos.q == 1
    assert pos.closed_lots == 1.0
    assert abs(pos.realized_gross - 0.005 * T.multiplier) < 1e-9


# ---------------------------------------------------------------- 时段/风控
def test_session_phase():
    sched = SessionSchedule()
    assert sched.phase(0.0, 900.0, 30.0) == PHASE_TRADING
    assert sched.phase(TRADING_SECONDS_PER_DAY - 500, 900.0, 30.0) == PHASE_FLATTEN
    assert sched.phase(TRADING_SECONDS_PER_DAY - 10, 900.0, 30.0) == PHASE_CLOSED


def test_risk_inventory_bounds():
    risk = RiskManager(T, RiskParams(max_inventory=5))
    risk.new_day(100.0)
    bid, ask = 99.995, 100.005
    assert risk.filter_quotes(bid, ask, 5) == (None, ask)     # 多头触界撤买
    assert risk.filter_quotes(bid, ask, -5) == (bid, None)    # 空头触界撤卖
    assert risk.filter_quotes(bid, ask, 0) == (bid, ask)
    # 价格带裁剪
    risk2 = RiskManager(T, RiskParams(max_inventory=5))
    risk2.new_day(100.0)
    lo, hi = 100.0 * (1 - T.price_limit_pct), 100.0 * (1 + T.price_limit_pct)
    b, a = risk2.filter_quotes(lo - 0.5, hi + 0.5, 0)
    assert lo - 1e-9 <= b <= hi + 1e-9
    assert lo - 1e-9 <= a <= hi + 1e-9


# ---------------------------------------------------------------- 回测不变量
def _small_backtest(mode=MODE_AS, days=1, **kw):
    market = MarketParams(price0=100.0, daily_sigma=0.30)
    strategy = StrategyParams(gamma=0.03, k_intensity=300.0)
    risk = RiskParams(max_inventory=10)
    backtest = BacktestParams(days=days, seed=7, sample_every=40)
    return Backtester(T, market, strategy, risk, backtest, mode=mode).run()


def test_backtest_invariants():
    res = _small_backtest()
    tick = T.tick_size
    max_inv = 10
    # 1) 每笔成交价在 tick 网格上
    for f in res.fills:
        assert abs(f.price / tick - round(f.price / tick)) < 1e-6
    # 2) 每日净持仓为 0（尾盘清仓 + 强平）
    for d in range(res.n_days):
        net = sum(f.side * f.lots for f in res.fills if f.day == d)
        assert abs(net) < 1e-9, f"第 {d+1} 日收盘未平: {net}"
    # 3) 库存样本不超限（瞬时可能到界，不可越界）
    for q in res.sample_inventory:
        assert abs(q) <= max_inv + 1e-9
    # 4) 指标可计算且有限
    m = compute_metrics(res)
    assert np.isfinite(m["total_pnl"])
    assert m["n_fills"] == len(res.fills)
    assert np.isfinite(m["avg_roundtrip_ticks"])


def test_backtest_paths_identical_across_modes():
    """同 seed 下不同策略模式的市场路径（中间价）必须一致。"""
    res_as = _small_backtest(MODE_AS)
    res_naive = _small_backtest(MODE_NAIVE)
    assert len(res_as.sample_mid) == len(res_naive.sample_mid)
    assert np.allclose(res_as.sample_mid, res_naive.sample_mid, atol=1e-12)
    assert np.allclose(res_as.sample_bb, res_naive.sample_bb, atol=1e-12)
    assert np.allclose(res_as.sample_ba, res_naive.sample_ba, atol=1e-12)


def test_backtest_has_fills_and_episodes():
    res = _small_backtest()
    assert len(res.fills) > 50, "单日成交数异常偏少"
    assert len(res.quote_episodes) > 100
    assert any(e.filled for e in res.quote_episodes)


# ---------------------------------------------------------------- 校准
def test_calibration_recovers_intensity():
    rng = np.random.default_rng(11)
    a_true, k_true = 0.8, 260.0
    episodes = []
    for _ in range(4000):
        delta = float(rng.uniform(0.0, 0.02))
        lam = a_true * math.exp(-k_true * delta)
        u = rng.exponential(1.0 / lam)
        w = 10.0  # 观测窗口
        if u < w:
            episodes.append(QuoteEpisode(0, "bid", delta, u, True))
        else:
            episodes.append(QuoteEpisode(0, "bid", delta, w, False))
    est = estimate_intensity(episodes)
    assert abs(est["A"] / a_true - 1) < 0.15
    assert abs(est["k"] / k_true - 1) < 0.15


# ---------------------------------------------------------------- 运行入口
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    sys.exit(1 if failed else 0)
