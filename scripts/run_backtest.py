#!/usr/bin/env python3
"""AS-MM 主入口：AS 做市策略回测 / 基准对比 / 参数扫描 / 强度校准。

用法示例：
    python scripts/run_backtest.py --contract T --days 10
    python scripts/run_backtest.py --contract TF --gamma 0.02 --no-sweep
    python scripts/run_backtest.py --contract T --sweep-days 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from as_mm import (  # noqa: E402
    BacktestParams,
    Backtester,
    MarketParams,
    RiskParams,
    StrategyParams,
    compute_metrics,
    daily_table,
    estimate_intensity,
    get_contract,
)
from as_mm.analytics import markouts  # noqa: E402
from as_mm.config import preset_for  # noqa: E402
from as_mm.plotting import (  # noqa: E402
    plot_daily_pnl,
    plot_day_snapshot,
    plot_equity,
    plot_quote_stats,
    plot_quote_window,
    plot_sweep,
)
from as_mm.risk import SessionSchedule  # noqa: E402
from as_mm.strategy import MODE_AS, MODE_NAIVE, MODE_NO_SKEW  # noqa: E402


def build_params(args):
    contract = get_contract(args.contract)
    preset = preset_for(args.contract)
    market: MarketParams = preset["market"]
    strategy: StrategyParams = preset["strategy"]
    if args.daily_sigma is not None:
        market.daily_sigma = args.daily_sigma
    if args.gamma is not None:
        strategy.gamma = args.gamma
    if args.k is not None:
        strategy.k_intensity = args.k
    if args.capture is not None:
        market.flow_capture_prob = args.capture
    if args.impact is not None:
        market.flow_impact = args.impact
    if args.queue_mean is not None:
        market.queue_mean = args.queue_mean
    risk = RiskParams(max_inventory=args.max_inventory)
    backtest = BacktestParams(days=args.days, seed=args.seed)
    return contract, market, strategy, risk, backtest


def run_one(contract, market, strategy, risk, backtest, mode, tag, outdir):
    t0 = time.time()
    res = Backtester(contract, market, strategy, risk, backtest, mode=mode).run()
    metrics = compute_metrics(res)
    metrics["mode"] = mode
    metrics["tag"] = tag
    metrics["runtime_sec"] = round(time.time() - t0, 1)
    if outdir:
        save_fills(res, os.path.join(outdir, f"{tag}_fills.csv"))
    print(f"  [{tag}] 完成：{metrics['n_fills']} 笔成交，"
          f"总盈亏 {metrics['total_pnl']:,.0f} 元（{metrics['runtime_sec']}s）")
    return res, metrics


def save_fills(res, path):
    sched = SessionSchedule()
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日", "时间", "方向", "价格", "手数", "手续费", "中间价", "类型"])
        for x in res.fills:
            w.writerow([
                x.day + 1, sched.wall_clock(x.ts),
                "买" if x.side > 0 else "卖", f"{x.price:.4f}", x.lots,
                f"{x.fee:.2f}", f"{x.mid:.4f}", x.note,
            ])


def save_samples(res, path):
    """保存 10 秒粒度采样序列：双边报价 / 持仓 / 累计盈亏（供外部细查）。"""
    sched = SessionSchedule()
    from as_mm.config import TRADING_SECONDS_PER_DAY
    tick = res.contract.tick_size
    day_base = 0.0
    cum = 0.0
    cur_day = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["日", "时间", "中间价", "我方买价", "我方卖价", "市场买价",
                    "市场卖价", "半价差(tick)", "报价偏移(tick)", "持仓(手)",
                    "日内累计盈亏(元)", "总累计盈亏(元)"])
        for t, mid, bid, ask, bb, ba, inv, eq in zip(
                res.sample_ts, res.sample_mid, res.sample_bid, res.sample_ask,
                res.sample_bb, res.sample_ba, res.sample_inventory,
                res.sample_equity):
            if int(t) != cur_day:
                cur_day = int(t)
                day_base = cum
            cum = eq
            half = (ask - bid) / 2 / tick if bid == bid and ask == ask else ""
            skew = (mid - (bid + ask) / 2) / tick if bid == bid and ask == ask else ""
            ts = (t - cur_day) * TRADING_SECONDS_PER_DAY
            w.writerow([
                cur_day + 1, sched.wall_clock(ts), f"{mid:.4f}",
                f"{bid:.4f}" if bid == bid else "", f"{ask:.4f}" if ask == ask else "",
                f"{bb:.4f}", f"{ba:.4f}",
                f"{half:.2f}" if half != "" else "", f"{skew:+.2f}" if skew != "" else "",
                int(inv), f"{eq - day_base:.2f}", f"{eq:.2f}",
            ])


def print_daily_detail(res):
    """逐日明细：盈亏 / 成交结构 / 费用 / 库存。"""
    print(f"\n  {'日':>3}  {'PnL(元)':>12}  {'成交':>6}（{'买':>5}/{'卖':>5}）"
          f"  {'费用(元)':>10}  {'库存末':>5}  {'库存峰':>5}")
    for row in daily_table(res):
        print(f"  {row['day']:>3}  {row['pnl']:>12,.0f}  {row['n_fills']:>6}"
              f"（{row['n_buys']:>5}/{row['n_sells']:>5}）"
              f"  {row['fees']:>10,.0f}  {row['end_inventory']:>5}"
              f"  {row['max_abs_inventory']:>5}")


def print_quote_excerpt(res, day=0, step=90, max_rows=12):
    """摘录某日双边报价采样（每 step 个采样点取一行，step=90 即约 15 分钟）。"""
    tick = res.contract.tick_size
    sched = SessionSchedule()
    from as_mm.config import TRADING_SECONDS_PER_DAY
    rows = [(t, m, b, a, q, e) for t, m, b, a, q, e in zip(
        res.sample_ts, res.sample_mid, res.sample_bid, res.sample_ask,
        res.sample_inventory, res.sample_equity) if int(t) == day]
    print(f"\n双边报价摘录（第 {day + 1} 日，10 秒粒度采样，每 15 分钟一行）：")
    print(f"  {'时间':>8}  {'中间价':>9}  {'我方买价':>9}  {'我方卖价':>9}"
          f"  {'半价差':>6}  {'偏移':>6}  {'持仓':>4}")
    for i in range(0, len(rows), step):
        if i // step >= max_rows:
            break
        t, m, b, a, q, _ = rows[i]
        ts = (t - day) * TRADING_SECONDS_PER_DAY
        half = (a - b) / 2 / tick
        skew = (m - (b + a) / 2) / tick
        print(f"  {sched.wall_clock(ts):>8}  {m:>9.4f}  {b:>9.4f}  {a:>9.4f}"
              f"  {half:>5.1f}t  {skew:>+5.1f}t  {int(q):>4}")


def print_fill_excerpt(res, day=0, n=8):
    """打印某日前 n 笔成交明细（含渠道）。"""
    sched = SessionSchedule()
    fills = [f for f in res.fills if f.day == day][:n]
    if not fills:
        return
    print(f"\n成交明细摘录（第 {day + 1} 日前 {len(fills)} 笔）：")
    print(f"  {'时间':>8}  {'方向':>4}  {'价格':>9}  {'中间价':>9}"
          f"  {'偏离(tick)':>8}  {'渠道':>10}")
    for f in fills:
        dev = (f.mid - f.price) / res.contract.tick_size
        print(f"  {sched.wall_clock(f.ts):>8}  {'买' if f.side > 0 else '卖':>4}"
              f"  {f.price:>9.4f}  {f.mid:>9.4f}  {dev:>+8.1f}  {f.rel:>10}")


def print_metrics(metrics, contract):
    tick_val = contract.tick_value
    print(f"""
━━━━━━━━━━━━━━ 绩效摘要（{metrics['tag']} / {metrics['mode']}） ━━━━━━━━━━━━━━
  总盈亏          {metrics['total_pnl']:>14,.0f} 元
  毛利（含费用）   {metrics['gross_pnl']:>14,.0f} 元
  手续费          {metrics['fees']:>14,.0f} 元（单笔均 {metrics['fees_per_fill']:,.1f} 元
                  = {metrics['fees_per_fill'] / tick_val:.2f} tick）
  年化 Sharpe     {metrics['sharpe_annual']:>14.2f}
  最大回撤        {metrics['max_drawdown']:>14,.0f} 元
  成交笔数        {metrics['n_fills']:>14,d}（做市 {metrics['n_mm_fills']:,}
                  清仓 {metrics['n_liquidation_fills']} 强平 {metrics['n_force_fills']}）
  回合数（手）    {metrics['closed_lots']:>14,.0f}
  回合毛价差      {metrics['avg_roundtrip_ticks']:>14.2f} tick
  回合净利        {metrics['net_per_roundtrip_yuan']:>14,.0f} 元/手
  平均库存        {metrics['avg_abs_inventory']:>14.1f} 手（峰值 {metrics['max_abs_inventory']:.0f}）
  平均占用保证金  {metrics['avg_margin_yuan']:>14,.0f} 元
  盈亏/保证金     {metrics['pnl_per_avg_margin']:>14.2%}
  挂单成交率      {metrics['quote_fill_rate']:>14.2%}
  Markout(tick)   1s {metrics['markout_ticks'].get('2', float('nan')):+.3f}
                  5s {metrics['markout_ticks'].get('10', float('nan')):+.3f}
                 30s {metrics['markout_ticks'].get('60', float('nan')):+.3f}
                120s {metrics['markout_ticks'].get('240', float('nan')):+.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")


def print_benchmark_table(all_metrics, tick_val):
    print("\n━━━━━━━━━━━ 策略消融对比（同市场路径） ━━━━━━━━━━━")
    header = (f"{'模式':<10}{'总盈亏':>12}{'毛利':>12}{'费用':>10}{'Sharpe':>8}"
              f"{'回撤':>10}{'成交':>8}{'回合tick':>9}{'净利/回合':>10}{'均库存':>8}")
    print(header)
    for m in all_metrics:
        print(f"{m['mode']:<10}{m['total_pnl']:>12,.0f}{m['gross_pnl']:>12,.0f}"
              f"{m['fees']:>10,.0f}{m['sharpe_annual']:>8.2f}{m['max_drawdown']:>10,.0f}"
              f"{m['n_fills']:>8,d}{m['avg_roundtrip_ticks']:>9.2f}"
              f"{m['net_per_roundtrip_yuan']:>10,.0f}{m['avg_abs_inventory']:>8.1f}")
    print("注：no_skew 仅保留 AS 最优价差、无库存偏移；naive 为固定 1 tick 报价。")


def main():
    ap = argparse.ArgumentParser(description="AS 国债期货做市策略回测")
    ap.add_argument("--contract", default="T", choices=["TS", "TF", "T", "TL"])
    ap.add_argument("--days", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gamma", type=float, default=None, help="有效风险厌恶 Gamma")
    ap.add_argument("--k", type=float, default=None, help="成交强度衰减系数")
    ap.add_argument("--max-inventory", type=int, default=20)
    ap.add_argument("--daily-sigma", type=float, default=None)
    ap.add_argument("--capture", type=float, default=None,
                    help="改善报价订单捕获概率")
    ap.add_argument("--impact", type=float, default=None, help="订单流冲击系数")
    ap.add_argument("--queue-mean", type=float, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--sweep-days", type=int, default=4)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    contract, market, strategy, risk, backtest = build_params(args)
    outdir = args.outdir or os.path.join("results", contract.symbol)
    os.makedirs(outdir, exist_ok=True)
    print(f"合约：{contract.name}（tick={contract.tick_size}，"
          f"乘数={contract.multiplier:,}，tick价值={contract.tick_value:.0f} 元）")
    print(f"参数：Gamma={strategy.gamma}，k={strategy.k_intensity}，"
          f"日波动率={market.daily_sigma}，库存上限={risk.max_inventory} 手，"
          f"{backtest.days} 个交易日，seed={backtest.seed}")

    # ---------- 1) AS 主回测 ----------
    print("\n[1/4] Avellaneda-Stoikov 主回测")
    res, metrics = run_one(contract, market, strategy, risk, backtest,
                           MODE_AS, f"{contract.symbol}_as", outdir)
    print_metrics(metrics, contract)
    mo = markouts(res)

    print_daily_detail(res)
    print_quote_excerpt(res, day=0)
    print_fill_excerpt(res, day=0)

    plot_equity(res, os.path.join(outdir, "equity_inventory.png"))
    plot_daily_pnl(res, os.path.join(outdir, "daily_pnl.png"))
    plot_quote_window(res, os.path.join(outdir, "quote_window.png"))
    plot_day_snapshot(res, os.path.join(outdir, "day_snapshot.png"), day=0)
    plot_quote_stats(res, os.path.join(outdir, "quote_stats.png"))
    save_samples(res, os.path.join(outdir, f"{contract.symbol}_as_samples.csv"))

    # ---------- 2) 成交强度校准 ----------
    print("\n[2/4] 成交强度校准（由主回测挂单记录估计 lambda(delta)=A*exp(-k*delta)）")
    est = estimate_intensity(res.quote_episodes)
    if not (est["A"] != est["A"]):  # 非NaN
        print(f"  估计 A = {est['A']:.3f} 次/秒，k = {est['k']:.1f} 1/价格"
              f"（样本 {est['n_episodes']} 段，成交 {est['n_filled']} 段）")
        for rng_, v in est.get("bins", {}).items():
            print(f"    delta {rng_}: 经验率 {v['empirical_rate']:.3f}/s "
                  f"模型率 {v['model_rate']:.3f}/s（n={v['n']}）")
    else:
        print(f"  {est.get('note', '样本不足')}")
    metrics["calibration"] = {kk: est.get(kk) for kk in ("A", "k", "n_episodes", "n_filled")}
    with open(os.path.join(outdir, f"{contract.symbol}_as_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    # ---------- 3) 基准消融对比 ----------
    print("\n[3/4] 基准消融对比")
    all_metrics = [metrics]
    for mode, tag in ((MODE_NO_SKEW, f"{contract.symbol}_noskew"),
                      (MODE_NAIVE, f"{contract.symbol}_naive")):
        _, m = run_one(contract, market, strategy, risk, backtest, mode, tag, outdir)
        all_metrics.append(m)
    print_benchmark_table(all_metrics, contract.tick_value)
    with open(os.path.join(outdir, "benchmarks.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)

    # ---------- 4) 参数扫描 ----------
    if not args.no_sweep:
        print(f"\n[4/4] Gamma x k 参数扫描（{args.sweep_days} 日/组）")
        gammas = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]
        ks = [100.0, 300.0, 1000.0]
        sweep_bp = BacktestParams(days=args.sweep_days, seed=backtest.seed)
        rows = []
        for g in gammas:
            for kk in ks:
                sp = StrategyParams(gamma=g, k_intensity=kk,
                                    vol_sample_interval=strategy.vol_sample_interval,
                                    vol_halflife=strategy.vol_halflife,
                                    max_spread_ticks=strategy.max_spread_ticks)
                _, m = run_one(contract, market, sp, risk, sweep_bp, MODE_AS,
                               f"sweep_g{g}_k{kk:.0f}", None)
                rows.append({"gamma": g, "k": kk, "total_pnl": m["total_pnl"],
                             "sharpe": m["sharpe_annual"], "n_fills": m["n_fills"],
                             "avg_inventory": m["avg_abs_inventory"],
                             "avg_rt_ticks": m["avg_roundtrip_ticks"]})
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, "sweep.csv"), index=False, encoding="utf-8-sig")
        plot_sweep(df, os.path.join(outdir, "sweep_pnl.png"), "total_pnl")
        best = df.loc[df["total_pnl"].idxmax()]
        print(f"  最优（按总盈亏）：Gamma={best['gamma']}, k={best['k']:g} -> "
              f"PnL {best['total_pnl']:,.0f} 元，Sharpe {best['sharpe']:.2f}，"
              f"均库存 {best['avg_inventory']:.1f} 手")
        piv = df.pivot(index="gamma", columns="k", values="total_pnl")
        print("\n  总盈亏（元）矩阵 [行=Gamma, 列=k]：")
        print(piv.to_string(float_format=lambda x: f"{x:>12,.0f}"))
    else:
        print("\n[4/4] 跳过参数扫描（--no-sweep）")

    print(f"\n所有结果已保存至 {outdir}/")


if __name__ == "__main__":
    main()
