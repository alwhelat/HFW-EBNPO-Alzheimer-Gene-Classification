"""
run_all_analysis.py -- Master runner for all HFW-NPO analysis scripts.
Executes scripts in sequence and prints a final summary. Run main_data4.py first.
"""

from __future__ import annotations
import time, sys, traceback, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

scripts = [
    ("Statistical Tests",        "statistical_tests.py"),
    ("Ablation Study",           "ablation_study.py"),
    ("Extended Metrics",         "metrics_extended.py"),
    ("Gene Analysis",            "gene_analysis.py"),
    ("Convergence Comparison",   "npo_convergence_comparison.py"),
]

results_log = []
total_start = time.time()

print("HFW-NPO — Full Analysis Suite  (D1 + D2 + D4/GSE122063)")

for label, script_name in scripts:
    script_path = ROOT / script_name
    print(f"\n{label}: running {script_name} ...")
    t0 = time.time()
    try:
        runpy.run_path(str(script_path), run_name="__main__")
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        results_log.append((label, "OK", elapsed))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results_log.append((label, f"FAIL: {e}", elapsed))

total_elapsed = time.time() - total_start

print("\nFINAL SUMMARY")
for label, status, t in results_log:
    icon = "OK" if status == "OK" else "!!"
    print(f"  [{icon}] {label:<30} {t:.1f}s  {status}")
print(f"\n  Total time: {total_elapsed:.1f}s")
print(f"\n  Output files:")
fig_dir = ROOT / "figures"
res_dir = ROOT / "results"
for d in [fig_dir, res_dir]:
    for f in sorted(d.glob("*")) if d.exists() else []:
        print(f"    {f.relative_to(ROOT)}")
