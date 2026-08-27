"""绩效分析：PnL 分解、风险指标、markout、FIFO 回合统计。"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .backtest import BacktestResult, Fill


def fifo_round_trips(fills: list[Fill]) -> tuple[float, float]:
    """FIFO 配对成交，返回 (closed_lots, realized_price_units)。

    realized_price_units 为每手价差之和（价格单位），乘合约乘数得金额。
    """
    buys: deque[tuple[float, float]] = deque()
    sells: deque[tuple[float, float]] = deque()
    closed = 0.0
    realized = 0.0
    for f in fills:
        lots, price = f.lots, f.price
        if f.side > 0:
            while lots > 1e-12 and sells:
                sp, sl = sells[0]
                take = min(lots, sl)
                realized += take * (sp - price)
                closed += take
                lots -= take
                if take >= sl - 1e-12:
                    sells.popleft()
                else:
                    sells[0] = (sp, sl - take)
            if lots > 1e-12:
                buys.append((price, lots))
        else:
            while lots > 1e-12 and buys:
                bp, bl = buys[0]
                take = min(lots, bl)
                realized += take * (price - bp)
                closed += take
                lots -= take
                if take >= bl - 1e-12:
                    buys.popleft()
                else:
                    buys[0] = (bp, bl - take)
            if lots > 1e-12:
                sells.append((price, lots))
    return closed, realized


def markouts(
    res: BacktestResult, horizons_steps: tuple[int, ...] = (2, 10, 60, 240)
) -> dict[int, float]:
    """成交后 markout（单位 tick，正=有利）。

    markout_h = side * (mid[t+h] - fill_price) / tick
    """
    tick = res.contract.tick_size
    out: dict[int, float] = {}
    for h in horizons_steps:
        vals = []
        for f in res.fills:
            arr = res.mid_full_days[f.day]
            j = f.step + h
            if j < len(arr):
                vals.append(f.side * (arr[j] - f.price) / tick)
        out[h] = float(np.mean(vals)) if vals else float("nan")
    return out


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if len(equity) else 0.0


def compute_metrics(res: BacktestResult) -> dict:
    """汇总回测绩效指标。"""
    c = res.contract
    daily = np.asarray(res.daily_pnl, dtype=float)
    total_pnl = float(daily.sum())
    fees = float(sum(f.fee for f in res.fills))
    n_fills = len(res.fills)

    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    sharpe = float(daily.mean() / std * np.sqrt(252)) if std > 0 else float("nan")

    eq = np.asarray(res.sample_equity, dtype=float)
    mdd = max_drawdown(eq)
    inv = np.asarray(res.sample_inventory, dtype=float)

    closed, realized = fifo_round_trips(res.fills)
    tick_val = c.tick_value
    avg_rt_ticks = realized / closed / c.tick_size if closed > 0 else float("nan")
    net_per_rt = (realized * c.multiplier - fees) / closed if closed > 0 else float("nan")

    notes = Counter(f.note for f in res.fills)
    mid_samples = np.asarray(res.sample_mid, dtype=float)
    margin_avg = float(np.mean(np.abs(inv)) * (np.nanmean(mid_samples) if len(mid_samples) else 0.0)
                       * c.multiplier * c.margin_rate)

    filled_eps = sum(1 for e in res.quote_episodes if e.filled)
    total_eps = len(res.quote_episodes)

    mo = markouts(res)

    return {
        "total_pnl": total_pnl,
        "gross_pnl": total_pnl + fees,
        "fees": fees,
        "n_days": res.n_days,
        "daily_pnl_mean": float(daily.mean()) if len(daily) else float("nan"),
        "daily_pnl_std": std,
        "sharpe_annual": sharpe,
        "max_drawdown": mdd,
        "n_fills": n_fills,
        "n_mm_fills": notes.get("mm", 0),
        "n_liquidation_fills": notes.get("liquidation", 0),
        "n_force_fills": notes.get("force", 0),
        "n_halt_fills": notes.get("halt", 0),
        "closed_lots": closed,
        "avg_roundtrip_ticks": avg_rt_ticks,
        "net_per_roundtrip_yuan": net_per_rt,
        "fees_per_fill": fees / n_fills if n_fills else float("nan"),
        "avg_abs_inventory": float(np.mean(np.abs(inv))) if len(inv) else float("nan"),
        "max_abs_inventory": float(np.max(np.abs(inv))) if len(inv) else float("nan"),
        "avg_margin_yuan": margin_avg,
        "pnl_per_avg_margin": total_pnl / margin_avg if margin_avg > 0 else float("nan"),
        "n_quote_episodes": total_eps,
        "quote_fill_rate": filled_eps / total_eps if total_eps else float("nan"),
        "markout_ticks": {str(h): v for h, v in mo.items()},
        "n_halts": len(res.halts),
    }


def daily_table(res: BacktestResult) -> list[dict]:
    """逐日明细（PnL / 成交数 / 费用）。"""
    days = sorted({f.day for f in res.fills} | set(range(res.n_days)))
    out = []
    for d in days:
        day_fills = [f for f in res.fills if f.day == d]
        out.append({
            "day": d + 1,
            "pnl": res.daily_pnl[d] if d < len(res.daily_pnl) else float("nan"),
            "n_fills": len(day_fills),
            "fees": sum(f.fee for f in day_fills),
        })
    return out
