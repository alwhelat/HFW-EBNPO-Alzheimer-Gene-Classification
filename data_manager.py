import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer


class DataManager:
    """Loads and preprocesses Alzheimer's gene expression CSV datasets."""

    LABEL_COLS = {'class', 'CLASS', 'label', 'Label', 'LABEL', 'diagnosis', 'Diagnosis'}
    NON_FEATURE_COLS = {
        'Source name 2', 'Accession', 'GSMID', 'Tissue',
        'sample', 'Sample', 'id', 'ID', 'gsm', 'GSM'
    }

    def __init__(self, file_path, label_column=None):
        self.file_path = file_path
        self.label_column = label_column

    def load_and_clean_data(self):
        df = pd.read_csv(self.file_path)

        # Identify label column
        if self.label_column is not None:
            label_col = self.label_column
        else:
            label_col = self._detect_label_column(df)

        y_raw = df[label_col].values
        df = df.drop(columns=[label_col])

        # Drop known non-feature string/id columns
        drop_cols = [c for c in df.columns
                     if c in self.NON_FEATURE_COLS or df[c].dtype == object]
        df = df.drop(columns=drop_cols, errors='ignore')

        feature_names = list(df.columns)
        X_raw = df.values.astype(np.float64)

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_raw)

        # Scale to [0, 1]
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_imputed)

        # Binarize for AD studies: AD=1, all others=0
        unique_labels = set(str(v).strip() for v in y_raw)
        ad_labels = {l for l in unique_labels if l.upper() == 'AD'}
        if ad_labels and len(unique_labels) > 2:
            y_binary = np.array([1 if str(v).strip().upper() == 'AD' else 0 for v in y_raw])
        else:
            le = LabelEncoder()
            y_binary = le.fit_transform(y_raw)

        return X, y_binary, feature_names

    def _detect_label_column(self, df):
        for col in df.columns:
            if col in self.LABEL_COLS:
                return col
        # Fallback: last column
        return df.columns[-1]
