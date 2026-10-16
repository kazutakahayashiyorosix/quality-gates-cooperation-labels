import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = json.load(open("results/raop_results.json")); S = R["summary"]
row = {(r["scenario"], r["procedure"]): r for r in R["rows"]}
bars = [
    ("Naive rule (AUC > 0.70)", row[("documented", "naive")]),
    ("Practitioner baseline", row[("documented", "practitioner")]),
    ("Gates, no feature metadata", row[("undeclared", "gates")]),
    ("Gates, documented metadata", row[("documented", "gates")]),
]
labels = [b[0] for b in bars] + ["Request-time features only"]
aucs = [b[1]["model_auc"] for b in bars] + [S["request_only_auc"]]
nleak = [len(b[1]["post_outcome_features_used"]) for b in bars] + [0]
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(6.4, 2.9))
y = list(range(len(labels)))[::-1]
for yi, a, k in zip(y, aucs, nleak):
    ax.barh(yi, a - 0.5, left=0.5, height=0.62, color=ORANGE if k else BLUE,
            hatch="//" if k else None, edgecolor="white", linewidth=0)
    txt = f"{a:.3f}" + (f"  ({k} post-outcome)" if k else "  (0 post-outcome)")
    ax.text(a + 0.006, yi, txt, va="center", fontsize=8, color=INK)
ax.axvline(S["activity_auc"], color=MUTED, ls="--", lw=1)
ax.text(S["activity_auc"] + 0.004, len(labels) - 0.45, f"activity baseline {S['activity_auc']:.3f}", fontsize=7.5, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5, color=INK)
ax.set_xlim(0.5, 1.22); ax.set_xticks([0.5,0.6,0.7,0.8,0.9,1.0]); ax.set_xlabel("Cross-validated AUC of the certified model", fontsize=8.5, color=INK)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=8)
plt.tight_layout(); plt.savefig("results/fig_raop.pdf"); plt.savefig("results/fig_raop.png", dpi=200)
