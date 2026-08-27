"""参数校准：

1) 成交强度 lambda(delta) = A * exp(-k * delta) 的极大似然估计，
   输入为挂单暴露记录（QuoteEpisode 列表，含删失样本）；
2) 由库存风险预算反解有效风险厌恶系数 Gamma。
"""
from __future__ import annotations

import math

import numpy as np

from .backtest import QuoteEpisode


def _profile_nll(k: float, deltas: np.ndarray, w: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """给定 k，最优 A 下的负对数似然。返回 (nll, A)。"""
    decay = np.exp(-k * deltas)
    sum_d = float(d.sum())
    sum_w_decay = float((w * decay).sum())
    if sum_w_decay <= 0 or sum_d <= 0:
        return float("inf"), 0.0
    a_hat = sum_d / sum_w_decay
    ll = float((d * (math.log(a_hat) - k * deltas)).sum() - a_hat * sum_w_decay)
    return -ll, a_hat


def estimate_intensity(episodes: list[QuoteEpisode]) -> dict:
    """从挂单暴露记录估计 (A, k)。

    似然：成交样本贡献 ln(lambda) - lambda*u，未成交样本贡献 -lambda*w
    （u 为成交等待时间，此处以该段暴露时长近似）。
    """
    deltas, w, d = [], [], []
    for e in episodes:
        if e.exposure <= 0:
            continue
        deltas.append(max(e.delta, 0.0))  # 改善到中间价以内按 delta=0 处理
        w.append(e.exposure)
        d.append(1.0 if e.filled else 0.0)
    if sum(d) < 5:
        return {"A": float("nan"), "k": float("nan"), "n_episodes": len(deltas),
                "n_filled": int(sum(d)),
                "note": "成交样本不足，无法校准"}

    deltas = np.asarray(deltas)
    w = np.asarray(w)
    d = np.asarray(d)

    # 对数网格搜索 + 局部细化
    ks = np.logspace(math.log10(10), math.log10(5000), 400)
    best = min(ks, key=lambda k: _profile_nll(k, deltas, w, d)[0])
    lo, hi = best / 1.5, best * 1.5
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if _profile_nll(m1, deltas, w, d)[0] < _profile_nll(m2, deltas, w, d)[0]:
            hi = m2
        else:
            lo = m1
    k_hat = 0.5 * (lo + hi)
    nll, a_hat = _profile_nll(k_hat, deltas, w, d)

    # 分箱诊断：各距离档的经验成交率
    bins = {}
    if len(deltas):
        edges = np.quantile(deltas, np.linspace(0, 1, 6))
        edges = np.unique(edges)
        if len(edges) >= 2:
            for i in range(len(edges) - 1):
                m = (deltas >= edges[i]) & (deltas < edges[i + 1] + 1e-12)
                if m.sum() > 0 and w[m].sum() > 0:
                    rate = d[m].sum() / w[m].sum()
                    bins[f"[{edges[i]:.4f},{edges[i+1]:.4f}]"] = {
                        "empirical_rate": float(rate),
                        "model_rate": float(a_hat * math.exp(-k_hat * float(deltas[m].mean()))),
                        "n": int(m.sum()),
                    }
    return {
        "A": a_hat,                      # 距中间价 0 处的成交强度（次/秒）
        "k": k_hat,                      # 衰减系数（1/价格单位）
        "n_episodes": len(deltas),
        "n_filled": int(sum(d)),
        "mean_exposure": float(w.mean()),
        "nll": nll,
        "bins": bins,
    }


def gamma_from_inventory_budget(
    sigma_per_sec: float,
    tau: float,
    q_max: int,
    tick: float,
    skew_ticks: float = 1.0,
) -> float:
    """库存打满时保留价偏移 skew_ticks 个 tick 反解 Gamma。"""
    if sigma_per_sec <= 0 or tau <= 0 or q_max <= 0:
        raise ValueError("sigma_per_sec/tau/q_max 必须为正")
    return skew_ticks * tick / (sigma_per_sec * sigma_per_sec * tau * q_max)
