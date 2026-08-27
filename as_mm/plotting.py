"""回测可视化（matplotlib Agg 后端，输出 PNG）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from .backtest import BacktestResult  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False


def plot_equity(res: BacktestResult, path: str) -> None:
    """权益曲线 + 库存轨迹。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    t = np.asarray(res.sample_ts)
    eq = np.asarray(res.sample_equity)
    inv = np.asarray(res.sample_inventory)

    axes[0].plot(t, eq, lw=1.2, color="#1a6feb")
    axes[0].set_title(f"{res.contract.symbol} Avellaneda-Stoikov Market-Making "
                      f"({res.n_days} days, mode={res.mode})")
    axes[0].set_ylabel("Cumulative equity (CNY)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, inv, lw=0.8, color="#d4380d")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Inventory (lots)")
    axes[1].set_xlabel("Trading day")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_daily_pnl(res: BacktestResult, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    daily = np.asarray(res.daily_pnl)
    colors = ["#1a6feb" if x >= 0 else "#d4380d" for x in daily]
    ax.bar(range(1, len(daily) + 1), daily, color=colors)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Daily PnL (CNY)")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("PnL (CNY)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_quote_window(
    res: BacktestResult, path: str, day: int = 0, t0: float = 9_000, t1: float = 9_600
) -> None:
    """绘制某日一段窗口内的中间价 / 我方报价 / 市场最优价。"""
    ts = np.asarray(res.sample_ts)
    m = (ts >= day + t0 / 16_200) & (ts <= day + t1 / 16_200)
    if m.sum() < 4:  # 窗口无样本则取当日中段
        m = (ts >= day + 0.4) & (ts <= day + 0.5)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = (ts[m] - day) * 16_200 / 60  # 分钟
    ax.plot(x, np.asarray(res.sample_mid)[m], color="black", lw=1.2, label="Mid")
    ax.plot(x, np.asarray(res.sample_bb)[m], color="#7f8c9b", lw=0.8, ls="--",
            label="Market best bid")
    ax.plot(x, np.asarray(res.sample_ba)[m], color="#7f8c9b", lw=0.8, ls=":",
            label="Market best ask")
    ax.plot(x, np.asarray(res.sample_bid)[m], color="#1a6feb", lw=0.8,
            marker=".", ms=2, label="Our bid")
    ax.plot(x, np.asarray(res.sample_ask)[m], color="#d4380d", lw=0.8,
            marker=".", ms=2, label="Our ask")
    ax.set_title(f"Day {day + 1} quoting behaviour "
                 f"(window {int(t0)}-{int(t1)} trading seconds)")
    ax.set_xlabel("Trading minute")
    ax.set_ylabel("Price (CNY)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sweep(df, path: str, value_col: str = "total_pnl") -> None:
    """参数扫描热力图（Gamma x k -> 指标）。"""
    gammas = sorted(df["gamma"].unique())
    ks = sorted(df["k"].unique())
    mat = np.full((len(gammas), len(ks)), np.nan)
    for _, row in df.iterrows():
        mat[gammas.index(row["gamma"]), ks.index(row["k"])] = row[value_col]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
    ax.set_yticks(range(len(gammas)), [f"{g:g}" for g in gammas])
    ax.set_xlabel("Intensity decay k (1/price)")
    ax.set_ylabel("Effective risk aversion Gamma")
    title = "Total PnL (CNY)" if value_col == "total_pnl" else "Annualized Sharpe"
    ax.set_title(f"Parameter sweep: {title}")
    for i in range(len(gammas)):
        for j in range(len(ks)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:,.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
