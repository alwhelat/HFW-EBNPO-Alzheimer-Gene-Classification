<div align="center">

# HFW-EBNPO
### A Hybrid Filter-Wrapper Enhanced Binary Nomadic People Optimizer<br>for High-Dimensional Alzheimer's Gene Expression Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Conference](https://img.shields.io/badge/ICSCCW-2026-6366f1?style=flat-square)](https://icsccw.com/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-f59e0b?style=flat-square)]()

<br>

> **Feature selection for Alzheimer's disease gene expression data**  
> combining a three-score hybrid filter, SMOTE balancing, EBNPO wrapper optimization,  
> and a soft-voting ensemble classifier — evaluated on four GEO datasets.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)

---

## Overview

Diagnosing Alzheimer's disease (AD) from gene expression microarray data is a high-dimensional classification problem — tens of thousands of probes, hundreds of samples. This project presents **HFW-EBNPO**, a four-phase pipeline:

| Phase | Module | Purpose |
|-------|--------|---------|
| **1 — Hybrid Filter** | `data_manager.py` | Score probes with Fisher ratio + Mutual Information + Welch *t*-test; retain top-*k* |
| **2 — SMOTE** | `main.py` | Adaptive intra-fold oversampling to handle class imbalance |
| **3 — EBNPO Wrapper** | `npo_optimizer.py` | Enhanced Binary NPO with Lévy flight, clan–family hierarchy, and hill-climbing to select the minimal discriminative subset |
| **4 — Ensemble** | `main.py` | Soft-voting classifier: Random Forest + Calibrated SVM + Logistic Regression |

The key algorithmic contributions of **EBNPO** over the standard NPO are:

- **Hierarchical Clan–Family Population** — structured diversity to escape local optima
- **Two-Phase Adaptive Lévy Flight** — stagnation-aware step-size scaling (β = 1.5)
- **Opposition-Based Learning (OBL) Initialization** — faster early convergence
- **Anti-Stagnation Restarts** — partial population restart when no improvement for *P* = 8 iterations
- **Hill-Climbing Post-Processing** — flip-one-bit local search on the deterministic best solution

---

## Architecture

```
Raw GEO Data (microarray .csv / .txt.gz)
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 1 · Hybrid Filter  (fitted on TRAIN only)        │
│                                                         │
│  ω = 0.40 × Fisher + 0.30 × Mutual-Info + 0.30 × Welch │
│  → retain top-k probes (k = 80 or 200)                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 2 · Adaptive SMOTE  (train split only)           │
│  k_neighbors = min(5, minority_count − 1)               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 3 · EBNPO Wrapper Optimisation                   │
│                                                         │
│  Population: n_clans × n_families agents                │
│  Fitness: f = α·(1 − BalAcc_inner) + (1−α)·|S|/K       │
│  Transfer: sigmoid + stochastic binarisation            │
│  Lévy flight, migration, restart, hill-climb            │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Phase 4 · Soft-Voting Ensemble  (on TEST split)        │
│                                                         │
│  RF(300) + SVC(rbf, prob=True) + LR(L2)                 │
│  → Accuracy · BalAcc · F1 · MCC · AUC-ROC              │
└─────────────────────────────────────────────────────────┘
         5-fold Stratified Cross-Validation (outer loop)
```

---

## Datasets

All datasets are publicly available on [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/).

| ID | GEO Accession | Tissue | Condition | Platform |
|----|--------------|--------|-----------|----------|
| D1 | [GSE33000](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33000) | Brain cortex | AD vs Control | Illumina HumanHT-12 |
| D2 | [GSE132903](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132903) | Brain (bulk RNA) | AD vs Control | Illumina HumanHT-12 |
| D3 | [GSE63060](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63060) | Peripheral blood | AD vs Control | Illumina HumanHT-12 |
| D4 | [GSE122063](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122063) | Brain (multi-region) | AD vs Control | Illumina HumanHT-12 |

> **Note:** Raw data files are not included in this repository due to size and redistribution restrictions.  
> Download each dataset directly from the GEO links above and place them in `dataset/`.

---

## Results

### Classification Performance (5-fold Stratified CV)

| Dataset | Accuracy | Balanced Acc | Sensitivity | Specificity | F1-Score | AUC-ROC | Genes Selected |
|---------|----------|-------------|-------------|-------------|----------|---------|----------------|
| GSE33000 (D1) | **84.4 ± 4.4%** | 84.4 ± 4.4% | 86.3 ± 6.8% | 82.5 ± 5.2% | 84.6 ± 4.7% | **0.863 ± 0.040** | ~90 / 47,293 |
| GSE132903 (D2) | **86.4 ± 3.9%** | 86.4 ± 3.9% | 87.1 ± 6.0% | 85.7 ± 5.1% | 86.5 ± 4.0% | **0.893 ± 0.042** | ~90 / 47,293 |
| GSE122063 (D4) | **97.5 ± 5.6%** | 97.5 ± 5.6% | 97.5 ± 5.6% | 97.5 ± 5.6% | 97.5 ± 5.6% | — | — |

### Filter Formula

```
ω(i) = 0.40 × Fisher(i)  +  0.30 × MutualInfo(i)  +  0.30 × Welch(i)
```

### Fitness Function

```
f(S) = α · (1 − BalancedAccuracy_inner)  +  (1 − α) · √(|S| / K)
```

where *α* = 0.8 balances classification quality against feature parsimony.

### Gene Enrichment Figures

GO/KEGG enrichment analysis on the selected gene sets (generated by `make_enrichment_figures.py`):

| Figure | Description |
|--------|-------------|
| `results/gene_symbols_enrichment/fig_enrichment_PANEL_manuscript.png` | Full manuscript panel |
| `results/gene_symbols_enrichment/fig_enrichment_heatmap.png` | Cross-dataset enrichment heatmap |
| `results/gene_symbols_enrichment/fig_enrichment_barchart.png` | Top GO terms bar chart |
| `results/gene_symbols_enrichment/fig_enrichment_bubble_GSE33000.png` | Bubble plot — D1 |
| `results/gene_symbols_enrichment/fig_enrichment_bubble_GSE132903.png` | Bubble plot — D2 |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/alwhelat/HFW-EBNPO-Alzheimer-Gene-Classification.git
cd HFW-EBNPO-Alzheimer-Gene-Classification

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Array operations |
| `pandas` | ≥ 2.0 | Data loading & manipulation |
| `scipy` | ≥ 1.11 | Lévy flight, statistical tests |
| `scikit-learn` | ≥ 1.3 | Classifiers, CV, metrics |
| `imbalanced-learn` | ≥ 0.11 | SMOTE oversampling |
| `openpyxl` | ≥ 3.1 | Excel output support |
| `matplotlib` | ≥ 3.7 | Figures & plots |
| `seaborn` | ≥ 0.12 | Statistical visualisations |
| `statsmodels` | ≥ 0.14 | Wilcoxon / Friedman tests |
| `tqdm` | ≥ 4.65 | Progress bars |

---

## Usage

### Run main pipeline (D1 — GSE33000 & D2 — GSE132903)

```bash
python main.py
```

### Run D3 — Blood dataset (GSE63060)

```bash
python run_experiments_gse63060.py
```

### Run full analysis suite (ablation, metrics, statistics, figures)

```bash
python run_all_analysis.py
```

### Individual modules

```bash
# Ablation study (component contribution)
python ablation_study.py

# Extended metrics (precision, recall, MCC, AUC)
python metrics_extended.py

# Statistical significance tests (Wilcoxon, Friedman)
python statistical_tests.py

# Convergence comparison plots
python npo_convergence_comparison.py

# GO/KEGG enrichment figures
python make_enrichment_figures.py

# Illumina probe → gene symbol mapping
python ilmn_to_genesymbol.py
```

---

## Project Structure

```
HFW-EBNPO-Alzheimer-Gene-Classification/
│
├── main.py                        ← Pipeline entry point (D1 / D2)
├── npo_optimizer.py               ← Core EBNPO optimizer
├── data_manager.py                ← Dataset loading & hybrid filter
├── data_loader_new.py             ← Alternative GEO data loader
├── gene_analysis.py               ← Gene stability & overlap analysis
├── ablation_study.py              ← Ablation experiments (C1–C9)
├── metrics_extended.py            ← Extended evaluation metrics
├── statistical_tests.py           ← Wilcoxon / Friedman significance tests
├── npo_convergence_comparison.py  ← Convergence curve plots
├── make_enrichment_figures.py     ← GO / KEGG enrichment visualisations
├── ilmn_to_genesymbol.py          ← Illumina probe → gene symbol mapping
├── run_all_analysis.py            ← Master runner (all modules)
│
├── dataset/                       ← GEO datasets (download separately)
│   ├── Data1(GSE33000).csv
│   ├── GES5281 and GSE48350.csv
│   ├── GSE122063_series_matrix.txt.gz
│   ├── GSE63060_series_matrix.txt.gz
│   ├── GSE63061_series_matrix.txt.gz
│   ├── GSE97760_series_matrix.txt.gz
│   └── GSE132903_RAW/
│
├── results/                       ← Experiment outputs (committed)
│   ├── Data1_GSE33000/            ← Per-fold gene lists & convergence
│   ├── Data2_GSE132903/           ← Per-fold gene lists & convergence
│   ├── gene_symbols_enrichment/   ← GO/KEGG enrichment figures & tables
│   ├── unified_project_results.json
│   ├── ablation_report.json
│   ├── extended_metrics.json
│   ├── statistical_report.json
│   └── convergence_report.json
│
├── requirements.txt               ← Python dependencies
├── LICENSE                        ← MIT License
└── README.md
```

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@inproceedings{alwhelat2026hfwebnpo,
  title     = {A Hybrid Filter-Wrapper Nomadic People Optimizer Framework
               for High-Dimensional Alzheimer's Gene Expression Classification},
  author    = {Alwhelat, Almuntadher and Abiyev, Rahib H.},
  booktitle = {Proceedings of the 13th International Conference on
               Computer Science, Computer Engineering, and Social Media (ICSCCW-2026)},
  year      = {2026},
  note      = {Under review}
}
```

A `CITATION.cff` file is also provided for automated citation tools (Zenodo, GitHub "Cite this repository").

---

## Authors

**Almuntadher Alwhelat**  
Near East University / Al-Farabi University, Baghdad, Iraq  
📧 [ 20235107@std.neu.edu.tr](mailto: 20235107@std.neu.edu.tr)
📧 [almuntadher.mahmood@alfarabiuc.edu.iq]
🔗 [github.com/alwhelat](https://github.com/alwhelat)

**Prof. Rahib H. Abiyev**  
Near East University, Nicosia, North Cyprus

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built for ICSCCW 2026 · Alzheimer's Disease Gene Expression Research</sub>
</div>
