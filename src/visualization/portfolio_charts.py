"""Build the three dashboard-style result images in portfolio/ from saved holdout results.

Run from the project root:  uv run python -m src.visualization.portfolio_charts
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "portfolio"
OUT.mkdir(exist_ok=True)

# palette (reference instance, light mode)
PAGE = "#f9f9f7"
SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "#e6e5e0"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAYFILL = "#d6d5ce"
GOOD = "#006300"
CRIT = "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
    "figure.facecolor": PAGE,
})

W, H, DPI = 16, 10, 125  # 2000 x 1250 px

hold = pd.read_csv(ROOT / "results/final_holdout/summary.csv")
sweep = pd.read_csv(ROOT / "results/final_holdout/holdout_threshold_sweep.csv")
lgb = hold[hold.model == "LightGBM"].iloc[0]


def card(fig, x, y, w, h):
    """Rounded white card in figure coords."""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                       transform=fig.transFigure, facecolor=SURF,
                       edgecolor=BORDER, linewidth=1.2, zorder=-10)
    fig.patches.append(p)


def header(fig, title, subtitle):
    fig.text(0.045, 0.935, title, fontsize=26, weight="bold", color=INK)
    fig.text(0.045, 0.895, subtitle, fontsize=14, color=INK2)


def footer(fig, note):
    fig.text(0.045, 0.03, note, fontsize=10.5, color=MUTED)


def style_ax(ax, grid_axis="y"):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=11)


def fmt(n):
    return f"{int(n):,}"


# ---------------------------------------------------------------- 1. overview
fig = plt.figure(figsize=(W, H), dpi=DPI)
header(fig, "Card Fraud Detection: Results",
       "LightGBM model scored on the latest 2 months of transactions it never saw during training")

tiles = [
    ("Transactions screened", fmt(lgb.holdout_rows), "2 months of unseen card payments"),
    ("Frauds caught", fmt(lgb.frauds_caught), f"{lgb.recall:.0%} of all {fmt(lgb.holdout_frauds)} fraud cases"),
    ("Alert precision", f"{lgb.precision:.0%}", "of flagged transactions were real fraud"),
    ("Alerts raised", f"{fmt(lgb.total_alerts)}", f"analyst review budget: {fmt(lgb.alert_budget)}"),
]
tx, gap = 0.045, 0.0125
tw = (0.91 - 3 * gap) / 4
for i, (label, value, sub) in enumerate(tiles):
    x = tx + i * (tw + gap)
    card(fig, x, 0.62, tw, 0.22)
    fig.text(x + 0.018, 0.795, label, fontsize=13, color=INK2)
    fig.text(x + 0.018, 0.69, value, fontsize=40, weight="bold", color=INK)
    fig.text(x + 0.018, 0.65, sub, fontsize=11, color=MUTED)
# status mark on workload tile
fig.text(tx + 3 * (tw + gap) + tw - 0.018, 0.795, "✓ Within budget",
         fontsize=12, color=GOOD, ha="right", weight="bold")

card(fig, 0.045, 0.08, 0.91, 0.5)
fig.text(0.063, 0.535, "Where the fraud and the alerts went", fontsize=16, weight="bold")
fig.text(0.063, 0.505, "Each bar is 100%. Blue is the model doing its job.", fontsize=12, color=INK2)

ax = fig.add_axes([0.2, 0.15, 0.72, 0.31])
style_ax(ax, grid_axis="x")
rows = [
    ("All fraud cases\n" + fmt(lgb.holdout_frauds),
     [(lgb.frauds_caught, "Caught", BLUE, "white"), (lgb.frauds_missed, "Missed", GRAYFILL, INK)]),
    ("All alerts raised\n" + fmt(lgb.total_alerts),
     [(lgb.frauds_caught, "Real fraud", BLUE, "white"), (lgb.false_alarms, "False alarm", ORANGE, "white")]),
]
for yi, (_, segs) in enumerate(rows):
    total = sum(s[0] for s in segs)
    left = 0
    for n, lab, col, txt in segs:
        share = n / total
        ax.barh(yi, share - 0.002, left=left + 0.001, height=0.55, color=col, edgecolor=SURF, linewidth=2)
        ax.text(left + share / 2, yi, f"{lab}\n{fmt(n)}  ({share:.0%})", ha="center", va="center",
                fontsize=13, color=txt, weight="bold")
        left += share
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=13, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.spines["bottom"].set_visible(False)

footer(fig, "Data: IEEE-CIS card transactions (590K rows). Model trained on the oldest 70%, tested once on the latest 30% (177,162 rows, 3.5% fraud).")
fig.savefig(OUT / "1_results_overview.png", facecolor=PAGE)
plt.close(fig)

# ------------------------------------------------------- 2. catch vs workload
fig = plt.figure(figsize=(W, H), dpi=DPI)
header(fig, "More Alerts Catch More Fraud, Up to What Analysts Can Review",
       "The alert threshold was set so the fraud team never gets more cases than it can handle")
card(fig, 0.045, 0.08, 0.91, 0.78)

s = sweep[sweep.model == "LightGBM"].sort_values("alerts")
s = s[s.alerts <= 30000]
ax = fig.add_axes([0.11, 0.17, 0.82, 0.62])
style_ax(ax)
budget = int(lgb.alert_budget)
ax.axvspan(0, budget, color="#eaf2fc", zorder=0)
ax.axvline(budget, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(budget + 300, 0.08, f"Review budget\n{fmt(budget)} alerts", fontsize=12, color=INK2, va="bottom")
ax.text(300, 0.97, "Within budget", fontsize=12, color=BLUE, weight="bold", va="top")
ax.plot(s.alerts, s.recall, color=BLUE, linewidth=2.5, solid_capstyle="round")
ax.scatter([lgb.total_alerts], [lgb.recall], s=140, color=BLUE, edgecolor=SURF, linewidth=2.5, zorder=5)
ax.annotate(f"Chosen operating point\n{fmt(lgb.total_alerts)} alerts → {lgb.recall:.0%} of fraud caught,\n"
            f"{lgb.precision:.0%} of alerts are real fraud",
            xy=(lgb.total_alerts, lgb.recall), xytext=(11000, 0.36), fontsize=13, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
ax.set_xlim(0, 30000)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v else "0"))
ax.set_xlabel("Alerts sent to analysts (on 177K transactions)", fontsize=13, labelpad=10)
ax.set_ylabel("Share of fraud caught", fontsize=13, labelpad=10)
footer(fig, "LightGBM on the 177,162-transaction holdout. Threshold chosen on earlier data and frozen before the holdout was scored.")
fig.savefig(OUT / "2_catch_rate_vs_workload.png", facecolor=PAGE)
plt.close(fig)

# --------------------------------------------------------- 3. model comparison
fig = plt.figure(figsize=(W, H), dpi=DPI)
header(fig, "Four Models Compared on the Same Unseen Data",
       "LightGBM ranked fraud best and was one of only two models that stayed within the alert budget")
card(fig, 0.045, 0.08, 0.44, 0.78)
card(fig, 0.515, 0.08, 0.44, 0.78)

order = ["LightGBM", "XGBoost", "Random Forest", "HistGradientBoosting"]
names = {"HistGradientBoosting": "HistGradBoost"}
h = hold.set_index("model").loc[order]
labels = [names.get(m, m) for m in order]
cols = [BLUE if m == "LightGBM" else GRAYFILL for m in order]

fig.text(0.063, 0.81, "Fraud ranking quality (PR-AUC)", fontsize=16, weight="bold")
fig.text(0.063, 0.78, "Higher is better. Random guessing scores 0.035.", fontsize=12, color=INK2)
ax = fig.add_axes([0.165, 0.15, 0.29, 0.58])
style_ax(ax, grid_axis="x")
ax.barh(range(4), h.pr_auc, color=cols, height=0.6, edgecolor=SURF, linewidth=2)
for i, v in enumerate(h.pr_auc):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=13, color=INK, weight="bold" if i == 0 else "normal")
ax.set_yticks(range(4), labels, fontsize=13, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, 0.6)
ax.spines["bottom"].set_visible(False)

fig.text(0.533, 0.81, "Alerts raised vs review budget", fontsize=16, weight="bold")
fig.text(0.533, 0.78, f"Budget: {fmt(budget)} alerts. Going over means fraud waits in a queue.", fontsize=12, color=INK2)
ax = fig.add_axes([0.635, 0.15, 0.29, 0.58])
style_ax(ax, grid_axis="x")
cap = 12000
for i, m in enumerate(order):
    a = h.loc[m, "total_alerts"]
    ok = bool(h.loc[m, "within_budget"])
    ax.barh(i, min(a, cap), color=cols[i], height=0.6, edgecolor=SURF, linewidth=2)
    mark, c = ("✓ within", GOOD) if ok else ("✗ over", CRIT)
    txt = f"{fmt(a)}" + ("  (6.6× budget)" if a > cap else "")
    xpos = max(min(a, cap), budget)
    if a > cap:
        ax.text(cap - 200, i, txt, va="center", ha="right", fontsize=12.5, color=INK, weight="bold",
                bbox=dict(boxstyle="square,pad=0.25", facecolor=GRAYFILL, edgecolor="none"))
        ax.text(cap - 200, i + 0.42, mark, va="center", ha="right", fontsize=11.5, color=c, weight="bold")
    else:
        ax.text(xpos + 200, i - 0.08, txt, va="center", fontsize=12.5, color=INK, weight="bold" if i == 0 else "normal")
        ax.text(xpos + 200, i + 0.2, mark, va="center", fontsize=11.5, color=c, weight="bold")
ax.axvline(budget, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(budget, -0.62, "budget", ha="center", fontsize=11, color=INK2)
ax.set_yticks(range(4), labels, fontsize=13, color=INK)
ax.invert_yaxis()
ax.set_xlim(0, cap)
ax.set_ylim(3.6, -0.8)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}K" if v else "0"))
ax.spines["bottom"].set_visible(False)
footer(fig, "Holdout of 177,162 transactions. XGBoost bar is cut at 12K: its retrained scores shifted scale, so the frozen threshold flagged 39,516 alerts.")
fig.savefig(OUT / "3_model_comparison.png", facecolor=PAGE)
plt.close(fig)
print("Saved:", *sorted(p.name for p in OUT.glob("*.png")))
