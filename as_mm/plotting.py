"""回测可视化（matplotlib Agg 后端，输出 PNG）。

图表分三层粒度：
- 全程：权益曲线 + 库存（plot_equity）、逐日盈亏（plot_daily_pnl）
- 单日：双边报价 / 持仓 / 日内盈亏联动快照（plot_day_snapshot）
- 微观：分钟级报价窗口 + 成交标记（plot_quote_window）
- 统计：报价宽度、库存偏移响应、持仓分布、日内盈亏热力（plot_quote_stats）
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from .backtest import BacktestResult  # noqa: E402
from .config import TRADING_SECONDS_PER_DAY  # noqa: E402
from .risk import SessionSchedule  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

_SCHED = SessionSchedule()


def _day_mask(res: BacktestResult, day: int) -> np.ndarray:
    ts = np.asarray(res.sample_ts)
    return (ts >= day) & (ts < day + 1)


def _minute_axis(ax, day: int) -> None:
    """交易分钟 -> 墙钟刻度（午休 11:30-13:00 压缩为一条分隔线）。"""
    total_min = TRADING_SECONDS_PER_DAY / 60  # 270
    marks = [(0, "9:15"), (60, "10:15"), (120, "11:15"),
             (121, "13:00"), (181, "14:00"), (240, "15:00"),
             (total_min, "15:15")]
    ax.set_xticks([m for m, _ in marks])
    ax.set_xticklabels([lab for _, lab in marks], fontsize=8)
    ax.axvline(120.5, color="gray", lw=0.7, ls="--", alpha=0.6)
    ax.set_xlim(0, total_min)


def plot_equity(res: BacktestResult, path: str) -> None:
    """权益曲线（含回撤区域）+ 库存轨迹（含 ±库存上限边界）。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    t = np.asarray(res.sample_ts)
    eq = np.asarray(res.sample_equity)
    inv = np.asarray(res.sample_inventory)

    axes[0].plot(t, eq, lw=1.2, color="#1a6feb", label="Equity")
    peak = np.maximum.accumulate(eq)
    axes[0].fill_between(t, eq, peak, color="#d4380d", alpha=0.15,
                         label="Drawdown")
    axes[0].set_title(f"{res.contract.symbol} Avellaneda-Stoikov Market-Making "
                      f"({res.n_days} days, mode={res.mode})")
    axes[0].set_ylabel("Cumulative equity (CNY)")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.3)

    lim = res.risk_params.max_inventory
    axes[1].fill_between(t, inv, 0, color="#d4380d", alpha=0.18, step="mid")
    axes[1].plot(t, inv, lw=0.6, color="#d4380d")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].axhline(lim, color="#b26a00", lw=0.9, ls="--", alpha=0.8)
    axes[1].axhline(-lim, color="#b26a00", lw=0.9, ls="--", alpha=0.8)
    axes[1].text(t[-1], lim, f" +{lim}", va="bottom", ha="right",
                 fontsize=8, color="#b26a00")
    axes[1].text(t[-1], -lim, f" -{lim}", va="top", ha="right",
                 fontsize=8, color="#b26a00")
    axes[1].set_ylabel("Inventory (lots)")
    axes[1].set_xlabel("Trading day")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_daily_pnl(res: BacktestResult, path: str) -> None:
    """逐日盈亏柱状图 + 累计盈亏折线（右轴）。"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    daily = np.asarray(res.daily_pnl)
    days = np.arange(1, len(daily) + 1)
    colors = ["#1a6feb" if x >= 0 else "#d4380d" for x in daily]
    ax.bar(days, daily, color=colors, width=0.65, label="Daily PnL")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Daily PnL (CNY)")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("PnL (CNY)")
    ax.grid(alpha=0.3, axis="y")

    ax2 = ax.twinx()
    ax2.plot(days, np.cumsum(daily), lw=1.6, color="#237804",
             marker="o", ms=3, label="Cumulative")
    ax2.set_ylabel("Cumulative PnL (CNY)", color="#237804")
    ax2.tick_params(axis="y", colors="#237804")
    ax.set_xticks(days[:: max(1, len(days) // 20)])

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_day_snapshot(res: BacktestResult, path: str, day: int = 0) -> None:
    """单日运行快照：中间价与成交 / 双边报价宽度与偏移 / 持仓 / 日内盈亏。

    四面板共享交易分钟横轴（墙钟标注，午休压缩），展示一天内
    报价 -> 成交 -> 持仓 -> 盈亏的完整传导链条。
    """
    tick = res.contract.tick_size
    m = _day_mask(res, day)
    if m.sum() < 4:
        return
    t = (np.asarray(res.sample_ts)[m] - day) * TRADING_SECONDS_PER_DAY / 60
    mid = np.asarray(res.sample_mid)[m]
    bid = np.asarray(res.sample_bid)[m]
    ask = np.asarray(res.sample_ask)[m]
    inv = np.asarray(res.sample_inventory)[m]

    # 日内累计盈亏：采样权益 - 当日开始前累计
    eq = np.asarray(res.sample_equity)[m]
    day_cum = eq - float(np.sum(res.daily_pnl[:day]))

    half = (ask - bid) / 2 / tick          # 半价差（tick）
    skew = (mid - (bid + ask) / 2) / tick  # 报价中枢偏移（tick，正=整体上移）

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.6, 2]})

    # 1) 中间价 + 成交标记
    ax = axes[0]
    ax.plot(t, mid, color="black", lw=1.0, label="Mid")
    bx = [f.ts / 60 for f in res.fills if f.day == day and f.side > 0]
    by = [f.price for f in res.fills if f.day == day and f.side > 0]
    sx = [f.ts / 60 for f in res.fills if f.day == day and f.side < 0]
    sy = [f.price for f in res.fills if f.day == day and f.side < 0]
    ax.scatter(bx, by, marker="^", s=14, color="#1a6feb", alpha=0.5,
               label=f"Buy fill ({len(bx)})")
    ax.scatter(sx, sy, marker="v", s=14, color="#d4380d", alpha=0.5,
               label=f"Sell fill ({len(sx)})")
    ax.set_ylabel("Price (CNY)")
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title(f"Day {day + 1} snapshot: price & fills / two-sided quotes / "
                 f"inventory / intraday PnL")

    # 2) 双边报价：半价差 + 中枢偏移
    ax = axes[1]
    ax.plot(t, half, lw=0.8, color="#1a6feb", label="Half-spread (ticks)")
    ax.plot(t, skew, lw=0.8, color="#d4380d", label="Quote skew (ticks)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Quote (ticks)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # 3) 持仓阶梯
    ax = axes[2]
    ax.fill_between(t, inv, 0, color="#d4380d", alpha=0.18, step="mid")
    ax.step(t, inv, where="mid", lw=0.8, color="#d4380d")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Inventory (lots)")
    ax.grid(alpha=0.3)

    # 4) 日内累计盈亏
    ax = axes[3]
    ax.plot(t, day_cum, lw=1.2, color="#237804")
    ax.axhline(0, color="gray", lw=0.8)
    ax.fill_between(t, day_cum, 0, where=day_cum >= 0, color="#237804", alpha=0.15)
    ax.fill_between(t, day_cum, 0, where=day_cum < 0, color="#d4380d", alpha=0.15)
    ax.set_ylabel("Intraday cum. PnL (CNY)")
    ax.set_xlabel("Wall clock")
    ax.grid(alpha=0.3)

    _minute_axis(axes[3], day)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_quote_window(
    res: BacktestResult, path: str, day: int = 0, t0: float = 60, t1: float = 360
) -> None:
    """绘制某日一段微观窗口：中间价 / 我方双边报价 / 市场最优价 / 成交标记。"""
    ts = np.asarray(res.sample_ts)
    m = (ts >= day + t0 / TRADING_SECONDS_PER_DAY) & \
        (ts <= day + t1 / TRADING_SECONDS_PER_DAY)
    if m.sum() < 4:  # 窗口无样本则取当日中段
        m = (ts >= day + 0.4) & (ts <= day + 0.5)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = (ts[m] - day) * TRADING_SECONDS_PER_DAY / 60
    ax.plot(x, np.asarray(res.sample_mid)[m], color="black", lw=1.2, label="Mid")
    ax.plot(x, np.asarray(res.sample_bb)[m], color="#7f8c9b", lw=0.8, ls="--",
            label="Market best bid")
    ax.plot(x, np.asarray(res.sample_ba)[m], color="#7f8c9b", lw=0.8, ls=":",
            label="Market best ask")
    ax.plot(x, np.asarray(res.sample_bid)[m], color="#1a6feb", lw=0.8,
            marker=".", ms=2, label="Our bid")
    ax.plot(x, np.asarray(res.sample_ask)[m], color="#d4380d", lw=0.8,
            marker=".", ms=2, label="Our ask")
    for f in res.fills:
        if f.day != day or not (t0 <= f.ts <= t1):
            continue
        marker, color = ("^", "#1a6feb") if f.side > 0 else ("v", "#d4380d")
        ax.plot(f.ts / 60, f.price, marker=marker, ms=7, ls="none",
                color=color, mec="black", mew=0.6, alpha=0.9)
    ax.set_title(f"Day {day + 1} quoting behaviour "
                 f"(window {int(t0)}-{int(t1)} trading seconds)")
    ax.set_xlabel("Trading minute")
    ax.set_ylabel("Price (CNY)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_quote_stats(res: BacktestResult, path: str) -> None:
    """微观结构统计四面板：半价差分布 / 库存-偏移响应 / 持仓分布 / 日内盈亏热力。"""
    tick = res.contract.tick_size
    mid = np.asarray(res.sample_mid, dtype=float)
    bid = np.asarray(res.sample_bid, dtype=float)
    ask = np.asarray(res.sample_ask, dtype=float)
    inv = np.asarray(res.sample_inventory, dtype=float)
    valid = ~(np.isnan(bid) | np.isnan(ask))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # 1) 半价差分布
    ax = axes[0, 0]
    half = (ask[valid] - bid[valid]) / 2 / tick
    ax.hist(half, bins=40, color="#1a6feb", alpha=0.75, edgecolor="white")
    ax.axvline(np.mean(half), color="#d4380d", lw=1.2,
               label=f"mean = {np.mean(half):.2f} ticks")
    ax.set_title("Quoted half-spread distribution")
    ax.set_xlabel("Half-spread (ticks)")
    ax.set_ylabel("Samples")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # 2) 报价中枢偏移 vs 库存（AS 库存规避响应）
    ax = axes[0, 1]
    skew = (mid[valid] - (bid[valid] + ask[valid]) / 2) / tick
    ax.scatter(inv[valid], skew, s=2, alpha=0.08, color="#d4380d")
    # 库存分桶均值线
    qs = np.percentile(inv[valid], np.linspace(5, 95, 19))
    centers, means = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        mm = (inv[valid] >= lo) & (inv[valid] < hi)
        if mm.sum() > 20:
            centers.append(np.mean(inv[valid][mm]))
            means.append(np.mean(skew[mm]))
    if centers:
        ax.plot(centers, means, lw=1.8, color="#1a6feb",
                label="bucket mean")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Quote skew vs inventory (AS inventory penalty)")
    ax.set_xlabel("Inventory (lots)")
    ax.set_ylabel("Skew (ticks, + = quote shifted up)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 3) 持仓分布
    ax = axes[1, 0]
    bins = np.arange(np.floor(inv.min()) - 0.5, np.ceil(inv.max()) + 1.5, 1.0)
    ax.hist(inv, bins=bins, color="#b26a00", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_title("Inventory distribution")
    ax.set_xlabel("Inventory (lots)")
    ax.set_ylabel("Samples")
    ax.grid(alpha=0.3, axis="y")

    # 4) 日内盈亏热力图（天 x 30 分钟时段）
    ax = axes[1, 1]
    n_days = res.n_days
    per_day = len(res.sample_equity) // n_days
    slot = per_day // 9  # 9 个 30 分钟时段
    if per_day and slot >= 2:
        eq = np.asarray(res.sample_equity, dtype=float)[:n_days * per_day] \
            .reshape(n_days, per_day)
        eq = np.hstack([np.zeros((n_days, 1)), eq])
        slot_pnl = eq[:, 1 + slot - 1::slot] - eq[:, :-1:slot]
        slot_pnl = slot_pnl[:, :9]
        im = ax.imshow(slot_pnl, cmap="RdYlGn", aspect="auto")
        labels = [_SCHED.wall_clock(i * 1800)[:5] for i in range(9)]
        ax.set_xticks(range(9), labels, fontsize=8)
        ax.set_yticks(range(n_days), [f"D{i + 1}" for i in range(n_days)],
                      fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.85, label="Slot PnL (CNY)")
    ax.set_title("Intraday PnL heatmap (30-min slots)")
    ax.set_xlabel("Slot start (wall clock)")
    ax.set_ylabel("Trading day")

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
