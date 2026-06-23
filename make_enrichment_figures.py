"""
Pathway Enrichment Figures for Alzheimer's Disease Manuscript
GSE33000 & GSE132903 — HFW-NPO Selected Biomarkers
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "gene_symbols_enrichment")
os.makedirs(OUT, exist_ok=True)

SOURCE_COLORS = {
    "GO:BP": "#3498DB",
    "KEGG":  "#E74C3C",
    "REAC":  "#2ECC71",
}
BLUE  = "#2E5FA3"
GREEN = "#2E8B57"
GREY  = "#7F8C8D"

gse33000_raw = [
    ("GO:BP","GO:0044248","Cellular catabolic process",2.37e-3,13,"ABCD3,CNOT7,FBXL8,FBXO38,FBXO8,GGT7,HEXB,HMBS,MADD,PNPLA3,PSME3,TRIM58,UBL3"),
    ("KEGG","KEGG:04110","Cell cycle",2.37e-3,5,"CCND3,CCNG1,CCNI,E2F6,RCC2"),
    ("KEGG","KEGG:05010","Alzheimer disease",2.37e-3,6,"ABCA7,HEXB,MADD,MAPKAPK3,SPTLC1,TFAM"),
    ("REAC","R-HSA-9612973","Autophagy",2.37e-3,7,"ABCD3,HEXB,MADD,PNPLA3,PSME3,TFRC,TRIM58"),
    ("REAC","R-HSA-5205685","PINK/Parkin Mitophagy",2.37e-3,4,"ATPAF2,MADD,TFAM,TIMM17A"),
    ("KEGG","KEGG:00860","Porphyrin metabolism",2.58e-3,3,"ALDH1L2,HMBS,MOCS2"),
    ("GO:BP","GO:0044257","Cellular protein catabolic process",3.09e-3,8,"FBXL8,FBXO38,FBXO8,GGT7,HEXB,MADD,PSME3,TRIM58"),
    ("REAC","R-HSA-3232118","SUMOylation of transcription cofactors",3.84e-3,4,"EWSR1,KLF11,TEAD2,ZNF41"),
    ("KEGG","KEGG:00600","Sphingolipid metabolism",4.27e-3,3,"CEPT1,HEXB,SPTLC1"),
    ("REAC","R-HSA-556833","Metabolism of lipids",5.73e-3,10,"ABCA7,ABCD3,AGPAT2,ALDH1L2,CEPT1,GPC1,HEXB,HSD17B6,PNPLA3,SPTLC1"),
    ("KEGG","KEGG:04010","MAPK signaling pathway",7.44e-3,6,"CCNG1,EFNA4,GRK5,MADD,MAPKAPK3,NRP2"),
    ("REAC","R-HSA-382551","Transport of small molecules",7.44e-3,8,"ABCA7,ABCD3,FXYD3,SLC22A13,SLC35F2,SLC38A6,SLC9A5,TFRC"),
    ("KEGG","KEGG:04151","PI3K-Akt signaling pathway",1.24e-2,6,"CCND3,CCNI,GPC1,IGFBP7,IL21R,NRP2"),
    ("GO:BP","GO:0007155","Cell adhesion",1.33e-2,8,"CNTNAP2,EFNA4,GPC1,IGFBP7,LRRC15,NRP2,SPON1,SPON2"),
    ("REAC","R-HSA-69278","Cell Cycle, Mitotic",1.47e-2,7,"CCND3,CCNG1,CCNI,E2F6,INO80E,RAD54L2,RCC2"),
    ("GO:BP","GO:0007268","Chemical synaptic transmission",2.62e-2,5,"CNTNAP2,MADD,SLC9A5,SPTLC1,STX18"),
    ("KEGG","KEGG:05016","Huntington disease",2.06e-2,4,"CNOT7,MADD,PSME3,TFAM"),
    ("REAC","R-HSA-1474244","ECM organization",2.42e-2,5,"GPC1,LRRC32,NRP2,SPON1,SPON2"),
    ("GO:BP","GO:0046034","ATP metabolic process",3.03e-2,3,"ATPAF2,MRPS34,TFAM"),
    ("GO:BP","GO:0006259","DNA metabolic process",3.40e-2,7,"DDX11,EWSR1,HIST1H2BJ,INO80E,RAD54L2,RCC2,TFAM"),
]

gse132903_raw = [
    ("KEGG","KEGG:04010","MAPK signaling pathway",6.97e-4,9,"CCNG1,EFNA4,FGFRL1,GRK5,ITGA5,MADD,MAPKAPK3,NRP2,RASAL2"),
    ("REAC","R-HSA-9612973","Autophagy",6.97e-4,9,"ABCD3,ATP6AP2,HEXB,MADD,PNPLA3,PSME3,SMCR8,TFRC,TRIM58"),
    ("REAC","R-HSA-2219528","PI3K-AKT signaling (Cancer)",1.48e-3,6,"GPC1,IL21R,ITGA5,NRP2,RASAL2,SHB"),
    ("KEGG","KEGG:05010","Alzheimer disease",4.22e-3,6,"ATP6AP2,HEXB,MADD,MAPKAPK3,SPTLC1,TFAM"),
    ("KEGG","KEGG:00860","Porphyrin metabolism",5.44e-3,3,"ALDH1A1,HMBS,MOCS2"),
    ("REAC","R-HSA-5205685","PINK/Parkin Mitophagy",5.44e-3,4,"ATPAF2,MADD,TFAM,TIMM17A"),
    ("GO:BP","GO:0042742","Defense response to bacterium",7.66e-3,7,"GBP4,HLA-DMA,IFNL1,IL21R,PSME3,SPON1,SPON2"),
    ("KEGG","KEGG:00600","Sphingolipid metabolism",7.66e-3,3,"CEPT1,HEXB,SPTLC1"),
    ("KEGG","KEGG:03030","DNA replication",7.66e-3,3,"DDX11,PCNA,RAD54L2"),
    ("REAC","R-HSA-3232118","SUMOylation of transcription cofactors",7.66e-3,4,"EWSR1,HDAC5,KLF11,ZNF41"),
    ("KEGG","KEGG:04640","Hematopoietic cell lineage",8.83e-3,4,"CD109,IL21R,ITGA5,TFRC"),
    ("REAC","R-HSA-1474244","ECM organization",8.83e-3,7,"GPC1,ITGA5,LRRC32,MYH9,NRP2,SPON1,SPON2"),
    ("KEGG","KEGG:05016","Huntington disease",1.43e-2,5,"CNOT7,HDAC5,MADD,PSME3,TFAM"),
    ("KEGG","KEGG:03460","Fanconi anemia pathway",1.30e-2,3,"DDX11,PCNA,RAD54L2"),
    ("GO:BP","GO:0044248","Cellular catabolic process",1.70e-2,12,"ABCD3,CNOT7,FBXL8,FBXO38,GGT7,HEXB,HMBS,MADD,PNPLA3,PSME3,TRIM58,UBL3"),
    ("GO:BP","GO:0007155","Cell adhesion",1.70e-2,9,"CNTNAP2,EFNA4,GPC1,ITGA5,LRRC15,MYH9,NRP2,SPON1,SPON2"),
    ("GO:BP","GO:0007268","Chemical synaptic transmission",2.74e-2,6,"ASIC1,CNTNAP2,MADD,SLC9A5,SPTLC1,STX18"),
    ("KEGG","KEGG:04151","PI3K-Akt signaling pathway",3.15e-2,6,"GPC1,IL21R,ITGA5,NRP2,RASAL2,SHB"),
    ("REAC","R-HSA-556833","Metabolism of lipids",3.74e-2,9,"ABCD3,ALDH1A1,CEPT1,GPC1,HEXB,HSD17B6,PLCB1,PNPLA3,SPTLC1"),
    ("REAC","R-HSA-73894","DNA Repair",3.74e-2,6,"DDX11,EWSR1,INO80E,PCNA,RAD54L2,TFAM"),
    ("REAC","R-HSA-913531","Interferon Signaling",4.80e-2,5,"ANPEP,GBP4,HLA-DMA,IFNL1,PSME3"),
]

def make_df(raw):
    df = pd.DataFrame(raw, columns=["source","term_id","term_name","p_value","gene_count","genes"])
    df["-log10p"] = -np.log10(df["p_value"])
    return df

df33  = make_df(gse33000_raw)
df132 = make_df(gse132903_raw)

cmap_heat = LinearSegmentedColormap.from_list("hm", ["#f7f7f7","#4393c3","#053061"])
palette = {"GSE33000": BLUE, "GSE132903": GREEN}


def bar_chart(df, title, ax, top=15):
    d = df.nsmallest(top, "p_value").sort_values("-log10p")
    colors = [SOURCE_COLORS.get(s, GREY) for s in d["source"]]
    bars = ax.barh(d["term_name"], d["-log10p"], color=colors,
                   edgecolor="white", linewidth=0.5, height=0.72)
    for bar, cnt in zip(bars, d["gene_count"]):
        ax.text(bar.get_width()+0.04, bar.get_y()+bar.get_height()/2,
                f"n={cnt}", va="center", fontsize=8, color="#2C3E50")
    ax.axvline(-np.log10(0.05), color="red", linestyle="--",
               linewidth=1, alpha=0.7, label="p=0.05")
    ax.set_xlabel("-log10(FDR p-value)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(axis='y', labelsize=8.5)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower right")


fig, axes = plt.subplots(1, 2, figsize=(20, 9))
bar_chart(df33,  "GSE33000 — Enriched Pathways\n(GPL6947, HumanHT-12 V3, n=83 genes)",  axes[0])
bar_chart(df132, "GSE132903 — Enriched Pathways\n(GPL10558, HumanHT-12 V4, n=102 genes)", axes[1])
src_handles = [mpatches.Patch(color=c, label=s) for s, c in SOURCE_COLORS.items()]
fig.legend(handles=src_handles, title="Database", loc="lower center",
           ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Pathway Enrichment Analysis of HFW-NPO Selected Biomarkers — Alzheimer's Disease",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_enrichment_barchart.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig_enrichment_barchart.png")


def bubble_plot(df, title, path, top=14):
    d = df.nsmallest(top, "p_value").sort_values("-log10p", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap_b = LinearSegmentedColormap.from_list("bp", ["#ffffcc","#fd8d3c","#800026"])
    norm = plt.Normalize(d["-log10p"].min(), d["-log10p"].max())
    sizes = (d["gene_count"] / d["gene_count"].max()) * 650 + 80
    sc = ax.scatter(d["-log10p"], range(len(d)), c=d["-log10p"],
                    cmap=cmap_b, norm=norm, s=sizes,
                    alpha=0.85, edgecolors="#2C3E50", linewidths=0.6, zorder=3)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["term_name"], fontsize=9)
    ax.set_xlabel("-log10(FDR p-value)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.colorbar(sc, ax=ax, shrink=0.5, label="-log10(p-value)")
    for sz, lbl in [(80,"n=1"),(300,"n=5"),(700,"n=10+")]:
        ax.scatter([], [], c=GREY, s=sz, alpha=0.7, label=lbl)
    ax.legend(title="Gene count", fontsize=8, loc="lower right")
    for i, (_, row) in enumerate(d.iterrows()):
        ax.text(0.015, i, row["source"], fontsize=7, color="white",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor=SOURCE_COLORS.get(row["source"], GREY), alpha=0.85),
                va="center", ha="left", transform=ax.get_yaxis_transform())
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {os.path.basename(path)}")

bubble_plot(df33,  "GSE33000 — Pathway Enrichment Bubble Plot\n(Alzheimer Brain, n=83 Biomarkers)",
            f"{OUT}/fig_enrichment_bubble_GSE33000.png")
bubble_plot(df132, "GSE132903 — Pathway Enrichment Bubble Plot\n(Alzheimer Brain, n=102 Biomarkers)",
            f"{OUT}/fig_enrichment_bubble_GSE132903.png")


shared_terms = [
    "Alzheimer disease","Autophagy","PINK/Parkin Mitophagy",
    "MAPK signaling pathway","Cell cycle","Sphingolipid metabolism",
    "Porphyrin metabolism","PI3K-Akt signaling pathway",
    "SUMOylation of transcription cofactors","Metabolism of lipids",
    "Cell adhesion","Chemical synaptic transmission",
    "Cellular catabolic process","Huntington disease","ECM organization",
]
rows = []
for term in shared_terms:
    for ds, df_ in [("GSE33000", df33), ("GSE132903", df132)]:
        r = df_[df_["term_name"]==term]
        if not r.empty:
            rows.append({"term":term,"dataset":ds,
                         "logp":r.iloc[0]["-log10p"],"n":r.iloc[0]["gene_count"],
                         "source":r.iloc[0]["source"]})
cdf = pd.DataFrame(rows)
terms_order = (cdf.groupby("term")["logp"].mean()
                  .sort_values(ascending=True).index.tolist())

fig, ax = plt.subplots(figsize=(13, 7))
for i, term in enumerate(terms_order):
    sub = cdf[cdf["term"]==term]
    for _, row in sub.iterrows():
        x = i + (-0.2 if row["dataset"]=="GSE33000" else 0.2)
        ax.scatter(x, row["logp"], s=row["n"]*40+80,
                   color=palette[row["dataset"]],
                   alpha=0.82, edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(x, row["logp"]+0.07, str(int(row["n"])),
                ha="center", va="bottom", fontsize=7.5, color="#2C3E50")

ax.set_xticks(range(len(terms_order)))
ax.set_xticklabels(terms_order, rotation=38, ha="right", fontsize=9.5)
ax.set_ylabel("-log10(FDR p-value)", fontsize=11)
ax.set_title("Comparative Pathway Enrichment — GSE33000 vs GSE132903\n"
             "HFW-NPO Biomarkers | Alzheimer's Disease",
             fontsize=13, fontweight="bold")
ax.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, alpha=0.5, label="p=0.05")
ax.spines[["top","right"]].set_visible(False)
ax.grid(axis="y", alpha=0.22)
ds_handles = [mpatches.Patch(color=c, label=ds, alpha=0.85) for ds, c in palette.items()]
ax.legend(handles=ds_handles, title="Dataset", fontsize=10, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_enrichment_dotplot_comparison.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig_enrichment_dotplot_comparison.png")


hm_terms = [
    ("Alzheimer disease","KEGG"),("Autophagy","REAC"),("PINK/Parkin Mitophagy","REAC"),
    ("MAPK signaling pathway","KEGG"),("Cell cycle","KEGG"),
    ("Sphingolipid metabolism","KEGG"),("Porphyrin metabolism","KEGG"),
    ("PI3K-Akt signaling pathway","KEGG"),("Metabolism of lipids","REAC"),
    ("Cell adhesion","GO:BP"),("Chemical synaptic transmission","GO:BP"),
    ("Cellular catabolic process","GO:BP"),("Huntington disease","KEGG"),
    ("ECM organization","REAC"),("SUMOylation of transcription cofactors","REAC"),
    ("DNA Repair","REAC"),("Interferon Signaling","REAC"),
    ("Transport of small molecules","REAC"),
]
mat = np.zeros((len(hm_terms), 2))
for i, (term, _) in enumerate(hm_terms):
    r33_  = df33[df33["term_name"]==term]
    r132_ = df132[df132["term_name"]==term]
    mat[i,0] = r33_.iloc[0]["-log10p"]  if not r33_.empty  else 0
    mat[i,1] = r132_.iloc[0]["-log10p"] if not r132_.empty else 0

fig, ax = plt.subplots(figsize=(7, 11))
im = ax.imshow(mat, cmap=cmap_heat, aspect="auto", vmin=0, vmax=3.5)
ax.set_xticks([0,1])
ax.set_xticklabels(["GSE33000\n(n=83)","GSE132903\n(n=102)"], fontsize=11)
ax.set_yticks(range(len(hm_terms)))
ax.set_yticklabels([t for t,_ in hm_terms], fontsize=9.5)
for i, (_, src) in enumerate(hm_terms):
    ax.text(-0.52, i, src, fontsize=8, va="center", ha="right",
            color=SOURCE_COLORS.get(src, GREY), fontweight="bold",
            transform=ax.get_yaxis_transform())
for i in range(len(hm_terms)):
    for j in range(2):
        v = mat[i,j]
        ax.text(j, i, f"{v:.2f}" if v>0 else "-",
                ha="center", va="center", fontsize=8.5,
                color="white" if v>1.8 else "#2C3E50")
ax.set_title("Pathway Enrichment Heatmap\n-log10(FDR p-value)",
             fontsize=12, fontweight="bold", pad=12)
plt.colorbar(im, ax=ax, shrink=0.4, label="-log10(p-value)")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_enrichment_heatmap.png", dpi=220,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig_enrichment_heatmap.png")


fig = plt.figure(figsize=(24, 21))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.38,
                      left=0.07, right=0.97, top=0.93, bottom=0.06)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[0, 2])
ax_d = fig.add_subplot(gs[1, :])

bar_chart(df33,  "A   GSE33000 — Enriched Pathways", ax_a, top=12)
bar_chart(df132, "B   GSE132903 — Enriched Pathways", ax_b, top=12)

im_c = ax_c.imshow(mat[:14,:], cmap=cmap_heat, aspect="auto", vmin=0, vmax=3.5)
ax_c.set_xticks([0,1])
ax_c.set_xticklabels(["GSE33000","GSE132903"], fontsize=9)
ax_c.set_yticks(range(14))
ax_c.set_yticklabels([t for t,_ in hm_terms[:14]], fontsize=8)
for i in range(14):
    for j in range(2):
        v = mat[i,j]
        ax_c.text(j, i, f"{v:.1f}" if v>0 else "-",
                  ha="center", va="center", fontsize=7.5,
                  color="white" if v>1.8 else "#2C3E50")
ax_c.set_title("C   Enrichment Heatmap (-log10 FDR p)", fontsize=11, fontweight="bold")
fig.colorbar(im_c, ax=ax_c, shrink=0.55)

for i, term in enumerate(terms_order):
    sub = cdf[cdf["term"]==term]
    for _, row in sub.iterrows():
        x = i + (-0.2 if row["dataset"]=="GSE33000" else 0.2)
        ax_d.scatter(x, row["logp"], s=row["n"]*42+80,
                     color=palette[row["dataset"]],
                     alpha=0.82, edgecolors="white", linewidths=0.8, zorder=3)
        ax_d.text(x, row["logp"]+0.07, str(int(row["n"])),
                  ha="center", va="bottom", fontsize=7.5, color="#2C3E50")
ax_d.set_xticks(range(len(terms_order)))
ax_d.set_xticklabels(terms_order, rotation=38, ha="right", fontsize=9.5)
ax_d.set_ylabel("-log10(FDR p-value)", fontsize=11)
ax_d.set_title("D   Cross-Dataset Pathway Comparison — GSE33000 vs GSE132903",
               fontsize=12, fontweight="bold")
ax_d.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, alpha=0.5)
ax_d.spines[["top","right"]].set_visible(False)
ax_d.grid(axis="y", alpha=0.2)
ax_d.legend(handles=ds_handles, title="Dataset", fontsize=9, loc="upper right")

fig.suptitle("Pathway Enrichment Analysis — HFW-NPO Selected Biomarkers\n"
             "Alzheimer's Disease (GSE33000 & GSE132903) | GO:BP · KEGG · Reactome",
             fontsize=15, fontweight="bold")
fig.legend(handles=src_handles, title="Database", loc="lower center",
           ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.005))
plt.savefig(f"{OUT}/fig_enrichment_PANEL_manuscript.png", dpi=250,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved: fig_enrichment_PANEL_manuscript.png")

print(f"\nAll figures saved to:\n{OUT}")
