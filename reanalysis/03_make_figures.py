"""Generates the README summary figures from the JSON results (no Kaggle data needed).
Usage: python reanalysis/03_make_figures.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
v1 = {r["name"]: r for r in json.load(open(os.path.join(HERE, "results", "results_v1_diagnosis.json"), encoding="utf-8"))}
v2 = json.load(open(os.path.join(HERE, "results", "results_v2.json"), encoding="utf-8"))

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)


# ---------- Figure 1: v1 (as published) vs v2 (corrected), PR-AUC with 95% bootstrap CI
rows = [
    ("24h · random 70/30 split", v1["A_original_24h"], v2["24h"]["split"]),
    ("48h · random 70/30 split", v1["A_original_48h"], v2["48h"]["split"]),
    ("24h · temporal validation", None, v2["24h"]["temporal"]),
    ("48h · temporal validation", None, v2["48h"]["temporal"]),
]
fig, ax = plt.subplots(figsize=(8.2, 3.9))
for i, (lab, a, b) in enumerate(rows):
    y = len(rows) - 1 - i
    if a:
        lo, hi = a["test_ap_ci"]
        ax.plot([lo, hi], [y + .17] * 2, color=ORANGE, lw=2, solid_capstyle="round")
        ax.plot(a["test_ap"], y + .17, "o", ms=9, color=ORANGE, mec=SURF, mew=2)
        ax.text(hi + .015, y + .17, f'{a["test_ap"]:.2f}', va="center", color=INK, fontsize=9)
    lo, hi = b["test_ap_ci"]
    ax.plot([lo, hi], [y - .17] * 2, color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(b["test_ap"], y - .17, "o", ms=9, color=BLUE, mec=SURF, mew=2)
    ax.text(hi + .015, y - .17, f'{b["test_ap"]:.2f}', va="center", color=INK, fontsize=9)
ax.axvline(0.027, color=GRAY, ls="--", lw=1)
ax.text(0.032, len(rows) - .45, "chance level\n(prevalence ≈ 0.03)", color=INK2, fontsize=8, va="top")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], color=INK)
ax.set_xlim(0, 1); ax.set_ylim(-.6, len(rows) - .3); ax.set_xlabel("PR-AUC on held-out patients (dot) · 95% bootstrap CI (line)")
ax.plot([], [], "o-", color=ORANGE, label="v1 · as published in the thesis (2025)")
ax.plot([], [], "o-", color=BLUE, label="v2 · corrected pipeline (reanalysis)")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("After fixing leakage and label construction, PR-AUC drops to ~0.2–0.3", loc="left", color=INK, fontsize=12)
style(ax); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig1_v1_vs_v2.png"), dpi=160); plt.close(fig)

# ---------- Figure 2: where the 48h result came from (each bar = one isolated fix)
steps = [
    ("v1 as published", v1["A_original_48h"]["test_ap"], ORANGE),
    ("− leaky feature\n(total_days_observed)", v1["B_48h_sem_total_days"]["test_ap"], ORANGE),
    ("− leaky feature\n+ fair negative cut-off", v1["C_48h_corte_aleatorio_neg"]["test_ap"], ORANGE),
    ("v2 corrected pipeline", v2["48h"]["split"]["test_ap"], BLUE),
]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
for i, (lab, val, c) in enumerate(steps):
    y = len(steps) - 1 - i
    ax.barh(y, val, height=.55, color=c)
    ax.text(val + .012, y, f"{val:.2f}", va="center", color=INK, fontsize=10)
ax.axvline(0.027, color=GRAY, ls="--", lw=1)
ax.set_yticks(range(len(steps))); ax.set_yticklabels([s[0] for s in steps][::-1], color=INK)
ax.set_xlim(0, 1); ax.set_xlabel("PR-AUC on held-out test patients, 48h horizon (first 3 bars: v1 dataset; last: rebuilt v2 dataset)")
ax.set_title("Where the published 0.75 came from: removing each flaw in turn", loc="left", color=INK, fontsize=12)
style(ax); fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "fig2_48h_flaw_decomposition.png"), dpi=160); plt.close(fig)
print("ok")
