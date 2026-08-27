# AS_MM

**Avellaneda-Stoikov (2008) 高频做市策略** — 面向境内国债期货市场（中金所 TS/TF/T/TL），以做市盈利最大化为目标。

## 概述

本策略以 Avellaneda-Stoikov 模型的近似解为基准：
- **保留价格** `r = P - q·Γ·σ²·τ`（库存偏移中枢）
- **最优总价差** `δ = Γ·σ²·τ + (2/Γ)·ln(1+Γ/k)`（单边半价差 δ/2）
- **双边报价** `bid = r - δ/2`, `ask = r + δ/2`，取整到 tick 网格

配套事件驱动回测引擎、市场仿真器（含订单流冲击逆向选择）、风控（库存界限 / 止损 / 尾盘清仓）、成交强度校准与绩效分析。支持三种模式消融对比：完整 AS、无库存偏移（no_skew）、朴素固定 1 tick（naive）。

## 安装

```bash
pip install numpy pandas matplotlib pytest
```

## 快速开始

```bash
# 10 日回测 + 消融对比 + 参数扫描
python scripts/run_backtest.py --contract T --days 10

# 仅主回测（跳过扫描）
python scripts/run_backtest.py --contract T --days 10 --no-sweep

# 自定义参数
python scripts/run_backtest.py --contract TF --gamma 0.01 --k 100 --days 10
```

结果保存至 `results/<合约>/`，含：
- `equity_inventory.png` — 权益曲线 + 库存轨迹
- `daily_pnl.png` — 逐日盈亏柱状图
- `quote_window.png` — 报价行为窗口
- `sweep_pnl.png` / `sweep.csv` — Gamma×k 扫描热力图
- `*_metrics.json` / `benchmarks.json` — 绩效指标
- `*_fills.csv` — 逐笔成交记录

## 回测绩效（T 主力，10 日，seed=7）

| 指标 | 数值 |
|---|---|
| 总盈亏 | +514,844 元（日均 +51,484） |
| 毛利 / 手续费 | +689,500 / 174,656 元 |
| 年化 Sharpe | 122.9 |
| 最大回撤 | 4,309 元 |
| 成交笔数 | 17,522（全部做市成交） |
| 回合毛价差 | 1.57 tick（净利 59 元/手） |
| 平均库存 | 0.7 手（峰值 7 手） |
| Markout | 1s +0.80、30s +0.74、120s +0.74 tick |

### 消融对比（同市场路径）

| 模式 | 总盈亏 | Sharpe | 回撤 | 均库存 |
|---|---|---|---|---|
| **AS（完整）** | +514,844 | 122.9 | 4,309 | 0.7 |
| no_skew | +793,439 | 14.1 | 231,549 | 10.9 |
| naive（固定1tick） | +503,408 | 7.9 | 257,353 | 16.1 |

> AS 完整版以最低的库存风险和回撤实现最高风险调整收益。no_skew 在本 seed 盈利更高但回撤 23 万、Sharpe 仅 14，跨 seed 表现极不稳定（见参数扫描）；naive 以高库存博取类似总盈亏，风险敞口最大。

### 成交渠道结构（最优参数 gamma=0.01, k=100）

| 渠道 | 占比 | markout30s | 说明 |
|---|---|---|---|
| touch（触价排队） | ~79% | +0.93 tick | 主要盈利来源 |
| improve（价差内改善） | ~5% | +0.62 tick | 竞争性成交 |
| marketable（被穿越） | ~16% | -0.58 tick | 逆向选择，需控制 |

## 参数预设

| 合约 | daily_sigma | gamma | k |
|---|---|---|---|
| TS（2年） | 0.08 | 0.008 | 100 |
| TF（5年） | 0.18 | 0.010 | 100 |
| T（10年） | 0.30 | 0.010 | 100 |
| TL（30年） | 0.55 | 0.012 | 100 |

参数由 Gamma×k 网格扫描（3 seeds 交叉验证）确定：盈利区在 gamma∈[0.005,0.02]×k∈[70,150]，峰值在 k≈100。校准模块反解 k≈161（经验成交率与模型 λ(δ)=A·e^(-kδ) 吻合度良好）。

## 目录结构

```
as_mm/
  model.py        # AS 核心公式 + EWMA 波动率估计
  market_sim.py   # 市场仿真：价格过程 + 订单簿 + 订单流 + 成交判定
  strategy.py     # 策略逻辑（三种模式）
  risk.py         # 风控：时段管理 / 库存界限 / 止损 / 尾盘清仓
  backtest.py     # 事件驱动回测引擎
  analytics.py    # 绩效：PnL 分解 / Sharpe / markout / 回合统计
  calibration.py  # 成交强度估计 + Gamma 反解
  config.py       # 全局参数与品种预设
  contracts.py    # 中金所国债期货合约规格
  plotting.py     # 可视化
scripts/
  run_backtest.py # 主入口
  diagnose_channels.py  # 成交渠道诊断
tests/
  test_as_mm.py   # 单元测试（13 项）
```

## 单元测试

```bash
python -m pytest tests/ -q
```

## 设计文档

详见 `docs/superpowers/specs/2026-08-27-avellaneda-stoikov-treasury-mm-design.md`。
