"""
gene_analysis.py -- Biomarker gene stability and overlap analysis for HFW-NPO paper.
Generates frequency heatmaps, Venn diagrams, and gene lists for pathway enrichment.
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
REP_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

_candidates = [
    ROOT / "results_data_4" / "unified_project_results.json",
    ROOT / "result-data 4"  / "unified_project_results.json",
    ROOT / "result 1"       / "unified_project_results.json",
    ROOT / "results"        / "unified_project_results.json",
]
RESULTS = next((p for p in _candidates if p.exists()), None)
if RESULTS is None:
    raise FileNotFoundError("unified_project_results.json not found. Run main_data4.py first.")
print(f"Loading results from: {RESULTS}")

with open(RESULTS, encoding="utf-8") as fh:
    data = json.load(fh)

gene_data = {}
for ds_name, ds in data["datasets"].items():
    short = ds_name.split("_")[1]
    fold_genes = [fold.get("selected_probe_ids", []) for fold in ds["folds"]]
    union_ids  = ds.get("final_outputs", {}).get("union_probe_ids", [])

    all_selected = [str(g) for fold in fold_genes for g in fold]
    freq = Counter(all_selected)
    n_folds = len(fold_genes)

    gene_data[short] = {
        "fold_genes": [[str(g) for g in fg] for fg in fold_genes],
        "union":      [str(g) for g in union_ids],
        "freq":       freq,
        "n_folds":    n_folds,
    }

ds_list  = list(gene_data.keys())
ds_cols  = {"GSE33000": "#378ADD", "GSE132903": "#1D9E75", "GSE122063": "#7F77DD"}

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 8))
fig1.suptitle("Gene Selection Frequency (Top-20 Biomarkers per Dataset)",
              fontsize=12, fontweight="bold")

top20_per_ds = {}
for ds, ax in zip(ds_list, axes1):
    freq    = gene_data[ds]["freq"]
    n_folds = gene_data[ds]["n_folds"]
    top20   = freq.most_common(20)
    top20_per_ds[ds] = top20

    genes   = [str(g) for g, _ in top20]
    counts  = [c for _, c in top20]
    pcts    = [c/n_folds*100 for c in counts]
    colors  = [ds_cols[ds] if p >= 60 else ("#EF9F27" if p >= 40 else "#B4B2A9") for p in pcts]
    y_pos   = range(len(genes)-1, -1, -1)

    bars = ax.barh(list(y_pos), pcts, color=colors, height=0.7)
    ax.set_yticks(list(y_pos))
    # Truncate long gene IDs for display
    gene_labels = [g[:18] + "…" if len(g) > 18 else g for g in genes]
    ax.set_yticklabels(gene_labels, fontsize=8)
    ax.set_xlabel("Selection Frequency (%)", fontsize=10)
    ax.set_title(ds, fontsize=11, fontweight="bold", color=ds_cols[ds])
    ax.set_xlim(0, 115)
    ax.axvline(x=60, color="#E24B4A", linestyle="--", linewidth=1,
               label="Stability threshold (60%)")
    ax.axvline(x=40, color="#EF9F27", linestyle="--", linewidth=0.8,
               label="Notable (40%)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, pct in zip(bars, pcts):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f"{pct:.0f}%", va="center", fontsize=8)
    if ds == ds_list[0]:
        ax.legend(fontsize=8, loc="lower right")

patches = [
    mpatches.Patch(color=ds_cols["GSE33000"],  label="≥60% stable"),
    mpatches.Patch(color="#EF9F27",             label="40-59% notable"),
    mpatches.Patch(color="#B4B2A9",             label="<40% occasional"),
]
fig1.legend(handles=patches, loc="lower center", ncol=3,
            fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
fig1.savefig(FIG_DIR / "fig_gene_stability.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"Saved: {FIG_DIR/'fig_gene_stability.png'}")

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 7))
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.set_aspect("equal"); ax2.axis("off")

union_sets = {ds: set(gene_data[ds]["union"]) for ds in ds_list}
sets = [union_sets[ds] for ds in ds_list]

A, B, C = sets[0], sets[1], sets[2]
AB   = A & B
AC   = A & C
BC   = B & C
ABC  = A & B & C
A_only = A - B - C
B_only = B - A - C
C_only = C - A - B

circles = [
    plt.Circle((3.5, 6.0), 2.4, alpha=0.25, color="#378ADD"),
    plt.Circle((6.5, 6.0), 2.4, alpha=0.25, color="#1D9E75"),
    plt.Circle((5.0, 3.5), 2.4, alpha=0.25, color="#7F77DD"),
]
for c in circles: ax2.add_patch(c)

label_pos = [
    (1.8, 7.5, ds_list[0], "#185FA5"),
    (8.2, 7.5, ds_list[1], "#0F6E56"),
    (5.0, 0.9, ds_list[2], "#3C3489"),
]
for x, y, lbl, c in label_pos:
    ax2.text(x, y, lbl, ha="center", va="center", fontsize=11,
             fontweight="bold", color=c)

count_pos = [
    (2.2, 5.8,  len(A_only),  "Only A"),
    (7.8, 5.8,  len(B_only),  "Only B"),
    (5.0, 2.3,  len(C_only),  "Only C"),
    (5.0, 6.7,  len(AB-ABC),  "A∩B"),
    (3.4, 4.2,  len(AC-ABC),  "A∩C"),
    (6.6, 4.2,  len(BC-ABC),  "B∩C"),
    (5.0, 5.2,  len(ABC),     "A∩B∩C"),
]
for x, y, n, lbl in count_pos:
    ax2.text(x, y, str(n), ha="center", va="center",
             fontsize=13, fontweight="bold", color="#2C2C2A")
    ax2.text(x, y-0.45, lbl, ha="center", va="center",
             fontsize=8, color="#5F5E5A")

ax2.set_title("Gene Union Set Overlap Across Datasets\n(union probe IDs from all folds)",
              fontsize=12, fontweight="bold", pad=15)
ax2.text(5.0, 0.2,
         f"Total unique: {len(A|B|C)} probes  |  Shared all 3: {len(ABC)} probes",
         ha="center", fontsize=10, color="#444441")
fig2.savefig(FIG_DIR / "fig_venn_genes.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {FIG_DIR/'fig_venn_genes.png'}")

for ds in ds_list:
    union = gene_data[ds]["union"]
    out   = REP_DIR / f"genes_for_enrichment_{ds}.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# HFW-NPO union gene list — {ds}\n")
        fh.write(f"# {len(union)} unique probes selected across 5 folds\n")
        fh.write("# Submit to: https://david.ncifcrf.gov or https://biit.cs.ut.ee/gprofiler/\n\n")
        for g in sorted(union):
            fh.write(f"{g}\n")
    print(f"Gene list saved: {out}")

report = {}
for ds in ds_list:
    freq    = gene_data[ds]["freq"]
    n_folds = gene_data[ds]["n_folds"]
    report[ds] = {
        "n_unique_genes":   len(gene_data[ds]["union"]),
        "n_folds":          n_folds,
        "stable_60pct":     [g for g, c in freq.items() if c/n_folds >= 0.60],
        "notable_40pct":    [g for g, c in freq.items() if 0.40 <= c/n_folds < 0.60],
        "top20_by_freq":    [{"gene": g, "count": c, "pct": round(c/n_folds*100,1)}
                             for g, c in freq.most_common(20)],
    }

sets_report = {
    "A_only (GSE33000)":   len(A_only), "B_only (GSE132903)":  len(B_only),
    "C_only (GSE122063)":  len(C_only), "AB_only":             len(AB-ABC),
    "AC_only":             len(AC-ABC), "BC_only":             len(BC-ABC),
    "ABC (all 3)":         len(ABC),    "total_union":         len(A|B|C),
    "ABC_genes":           list(ABC),
}
report["venn_overlap"] = sets_report

with open(REP_DIR / "gene_analysis.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
print(f"Saved: {REP_DIR/'gene_analysis.json'}")

print("\n=== GENE ANALYSIS SUMMARY ===")
for ds in ds_list:
    r = report[ds]
    print(f"\n{ds}: {r['n_unique_genes']} union probes  "
          f"| stable≥60%: {len(r['stable_60pct'])}  "
          f"| notable≥40%: {len(r['notable_40pct'])}")
print(f"\nShared by ALL 3 datasets: {len(ABC)} probes → {list(ABC)[:5]}...")
