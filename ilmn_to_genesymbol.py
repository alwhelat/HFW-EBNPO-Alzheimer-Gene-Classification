"""
ILMN Probe IDs -> Gene Symbols
Converts Illumina HumanHT-12 probe IDs to HGNC Gene Symbols.
Primary: Ensembl BioMart. Fallback: g:Profiler Convert API.
"""

import requests
import json
import time
import os

BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="illumina_humanht_12_v4" value="{probes}"/>
    <Attribute name="illumina_humanht_12_v4"/>
    <Attribute name="hgnc_symbol"/>
    <Attribute name="ensembl_gene_id"/>
  </Dataset>
</Query>"""

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/convert/"


def read_ilmn_ids(filepath):
    ids = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ILMN_"):
                ids.append(line)
    return ids


def query_biomart(probe_ids, batch_size=200):
    mapping = {}
    for i in range(0, len(probe_ids), batch_size):
        batch = probe_ids[i:i + batch_size]
        xml = BIOMART_XML.format(probes=",".join(batch))
        print(f"  BioMart batch {i//batch_size+1}: {len(batch)} probes...")
        for attempt in range(3):
            try:
                r = requests.post(BIOMART_URL, data={"query": xml}, timeout=60)
                r.raise_for_status()
                for line in r.text.strip().split("\n"):
                    parts = line.split("\t")
                    if len(parts) >= 2 and parts[0] and parts[1]:
                        mapping[parts[0].strip()] = parts[1].strip()
                break
            except Exception as e:
                print(f"    attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
        time.sleep(1)
    return mapping


def query_gprofiler(probe_ids):
    mapping = {}
    batch_size = 500
    for i in range(0, len(probe_ids), batch_size):
        batch = probe_ids[i:i + batch_size]
        print(f"  g:Profiler batch {i//batch_size+1}: {len(batch)} probes...")
        try:
            payload = {
                "organism": "hsapiens",
                "query": batch,
                "target": "HGNC",
                "numeric_ns": "ENTREZGENE_ACC"
            }
            r = requests.post(GPROFILER_URL, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            for entry in data.get("result", []):
                incoming = entry.get("incoming", "")
                converted = entry.get("converted", "")
                if incoming.startswith("ILMN_") and converted and converted != "None":
                    mapping[incoming] = converted
        except Exception as e:
            print(f"    g:Profiler failed: {e}")
        time.sleep(1)
    return mapping


def convert_file(input_path, output_path, dataset_name):
    print(f"\nDataset : {dataset_name}")
    print(f"Input   : {input_path}")
    print(f"Output  : {output_path}")

    probe_ids = read_ilmn_ids(input_path)
    print(f"Found {len(probe_ids)} ILMN probe IDs")

    print("\nStep 1: Querying Ensembl BioMart...")
    mapping = query_biomart(probe_ids)
    print(f"  BioMart mapped: {len(mapping)}/{len(probe_ids)}")

    unmapped_ids = [p for p in probe_ids if p not in mapping]
    if unmapped_ids:
        print(f"\nStep 2: Fallback to g:Profiler for {len(unmapped_ids)} unmapped probes...")
        gp_mapping = query_gprofiler(unmapped_ids)
        mapping.update(gp_mapping)
        print(f"  g:Profiler added: {len(gp_mapping)} more")

    converted = [(p, mapping[p]) for p in probe_ids if p in mapping]
    unmapped  = [p for p in probe_ids if p not in mapping]

    print(f"\nFinal: {len(converted)}/{len(probe_ids)} mapped, {len(unmapped)} unmapped")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Dataset: {dataset_name}\n")
        f.write(f"# Mapped: {len(converted)}/{len(probe_ids)} probes\n")
        f.write(f"# Sources: Ensembl BioMart + g:Profiler\n")
        f.write(f"# Ready for: DAVID (https://david.ncifcrf.gov) | g:Profiler (https://biit.cs.ut.ee/gprofiler/)\n#\n")
        for _, sym in converted:
            f.write(sym + "\n")

    table_path = output_path.replace(".txt", "_probe_mapping.txt")
    with open(table_path, "w") as f:
        f.write(f"# Probe-to-Gene mapping — {dataset_name}\n")
        f.write("ILMN_Probe\tGene_Symbol\n")
        for pid, sym in converted:
            f.write(f"{pid}\t{sym}\n")
        if unmapped:
            f.write("\n# Unmapped probes (submit ILMN IDs directly to DAVID):\n")
            for pid in unmapped:
                f.write(f"{pid}\tNOT_FOUND\n")

    print(f"Saved: {output_path}")
    print(f"Saved: {table_path}")
    return len(converted), len(probe_ids)


def main():
    base         = os.path.dirname(os.path.abspath(__file__))
    results_dir  = os.path.join(base, "results")
    out_dir      = os.path.join(results_dir, "gene_symbols_enrichment")
    os.makedirs(out_dir, exist_ok=True)

    tasks = [
        {
            "input":  os.path.join(results_dir, "genes_for_enrichment_GSE33000.txt"),
            "output": os.path.join(out_dir, "GSE33000_gene_symbols.txt"),
            "name":   "GSE33000 — Illumina HumanHT-12",
        },
        {
            "input":  os.path.join(results_dir, "genes_for_enrichment_GSE132903.txt"),
            "output": os.path.join(out_dir, "GSE132903_gene_symbols.txt"),
            "name":   "GSE132903 — Illumina HumanHT-12",
        },
    ]

    print("ILMN Probe IDs -> Gene Symbols Converter")
    print("BioMart primary  |  g:Profiler fallback")

    summary = []
    for t in tasks:
        if not os.path.exists(t["input"]):
            print(f"\nSkipping (file not found): {t['input']}")
            continue
        n, total = convert_file(t["input"], t["output"], t["name"])
        summary.append((t["name"], n, total))

    print("\nSUMMARY")
    for name, n, total in summary:
        pct = 100 * n / total if total else 0
        print(f"  {name}: {n}/{total} ({pct:.1f}%) mapped")
    print(f"\nOutput folder: {out_dir}")
    print("\nNext steps:")
    print("  DAVID     -> Upload gene symbol files to https://david.ncifcrf.gov")
    print("  g:Profiler -> Upload to https://biit.cs.ut.ee/gprofiler/gost")


if __name__ == "__main__":
    main()
