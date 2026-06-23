import io
import gzip
import numpy as np
import pandas as pd

def load_gse122063_from_gz(file_path):
    """Load GSE122063 series matrix from a gzipped file, returning expression matrix and labels."""
    print(f"Opening: {file_path}")

    metadata_lines = []
    data_lines = []
    in_data_section = False

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                in_data_section = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break

            if in_data_section:
                data_lines.append(line)
            else:
                metadata_lines.append(line)

    sample_ids = []
    characteristics_lines = []

    for line in metadata_lines:
        parts = [p.strip('"') for p in line.strip().split("\t")]
        if not parts or parts[0] == "":
            continue

        if parts[0] == "!Sample_geo_accession":
            sample_ids = parts[1:]
        elif parts[0] == "!Sample_characteristics_ch1":
            characteristics_lines.append(parts[1:])

    print(f"Found {len(sample_ids)} samples in metadata.")

    labels = None
    for c_line in characteristics_lines:
        if any("disease state:" in str(item).lower() for item in c_line):
            labels = [str(item).replace("disease state: ", "").strip() for item in c_line]
            break

    if labels is None and len(characteristics_lines) > 0:
        labels = characteristics_lines[0]

    if labels is None or len(labels) == 0:
        raise ValueError("Could not extract diagnosis labels from the file.")

    if len(sample_ids) != len(labels):
        print(f"Warning: sample/label count mismatch ({len(sample_ids)} vs {len(labels)}), trimming to minimum.")
        min_len = min(len(sample_ids), len(labels))
        sample_ids = sample_ids[:min_len]
        labels = labels[:min_len]

    df_meta = pd.DataFrame({"Sample": sample_ids, "Group": labels})

    def clean_group(val):
        v = str(val).lower().strip()
        if "control" in v or v == "non-demented controls":
            return 0
        elif "alzheimer" in v or "ad " in v or v == "ad":
            return 1
        return None

    df_meta['Target'] = df_meta['Group'].apply(clean_group)
    df_meta = df_meta.dropna(subset=['Target'])

    keep_samples = df_meta["Sample"].tolist()
    sample_to_target = dict(zip(df_meta["Sample"], df_meta["Target"].astype(int)))

    print(f"Class distribution: {dict(df_meta['Target'].value_counts())}  (0: Control, 1: AD)")

    print("Building expression matrix...")
    df_expr = pd.read_csv(io.StringIO("".join(data_lines)), sep="\t", index_col=0)

    available_samples = [s for s in keep_samples if s in df_expr.columns]
    df_expr = df_expr[available_samples]

    if df_expr.isna().sum().sum() > 0:
        df_expr = df_expr.fillna(df_expr.mean(axis=1))

    X = df_expr.T.values
    y = np.array([sample_to_target[s] for s in available_samples])
    gene_names = df_expr.index.tolist()

    print(f"Done. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, gene_names

if __name__ == "__main__":

    my_file_path = r"D:\MUN\PHD\PhD Thesis\retha\retha1\almuntadher-alwhelat\dataset\GSE122063_series_matrix.txt.gz"

    X, y, genes = load_gse122063_from_gz(my_file_path)

    print("\nFirst 10 target labels (y):", y[:10])
