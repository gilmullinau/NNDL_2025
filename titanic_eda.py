#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Titanic-style EDA script (enhanced, final)

Usage examples:
  # 1) Run on your own CSV
  python titanic_eda.py --input path/to/data.csv --target survived --output eda_report

  # 2) Auto-download Titanic from Kaggle and run on train.csv
  python titanic_eda.py --download-kaggle --output eda_report

  # 3) Download and run baseline model, too
  python titanic_eda.py --download-kaggle --baseline --output eda_report

Notes:
- Requires Kaggle API credentials if using --download-kaggle.
- Plots use matplotlib only (no seaborn), one chart per figure.
"""

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------
# Utilities
# ---------------------------

def snake_case(s: str) -> str:
    s = s.strip().replace('-', '_').replace(' ', '_')
    s = re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()
    s = re.sub(r'__+', '_', s)
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake_case(c) for c in df.columns]
    return df


def guess_id_columns(cols: List[str]) -> List[str]:
    candidates = {'passengerid', 'passenger_id', 'id'}
    return [c for c in cols if c in candidates]


def infer_feature_types(df: pd.DataFrame, target: str = None) -> Tuple[List[str], List[str]]:
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for c in df.columns:
        if target is not None and c == target:
            continue
        if c in guess_id_columns(df.columns):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Treat pclass as categorical even if numeric
            if c == 'pclass':
                cat_cols.append(c)
            else:
                num_cols.append(c)
        else:
            cat_cols.append(c)
    return cat_cols, num_cols


def ensure_output_dir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


# ---------------------------
# Overview & Missingness
# ---------------------------

def basic_overview(df: pd.DataFrame, outdir: Path):
    overview = {
        'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'duplicate_rows': int(df.duplicated().sum())
    }
    save_json(overview, outdir / 'overview.json')

    # Numeric describe — only if there are numeric cols
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        desc_num = df[num_cols].describe().T
    else:
        desc_num = pd.DataFrame()
    desc_num.to_csv(outdir / 'describe_numeric.csv')

    # Categorical describe — only if there are object/category cols
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        desc_cat = df[cat_cols].describe().T
    else:
        desc_cat = pd.DataFrame()
    desc_cat.to_csv(outdir / 'describe_categorical.csv')


def plot_missingness(df: pd.DataFrame, outdir: Path):
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure()
    miss.plot(kind='bar')
    plt.title('Missingness by column (fraction)')
    plt.ylabel('Fraction missing')
    plt.tight_layout()
    plt.savefig(outdir / 'missingness_by_column.png', dpi=150)
    plt.close()
    miss.to_csv(outdir / 'missingness_by_column.csv', header=['fraction_missing'])


# ---------------------------
# Target & Bivariate
# ---------------------------

def target_analysis(df: pd.DataFrame, target: str, outdir: Path):
    if target not in df.columns:
        return
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.to_numeric(y, errors='coerce')
    vc = y.value_counts(dropna=False)
    vc.to_csv(outdir / f'target_distribution.csv', header=['count'])
    plt.figure()
    vc.plot(kind='bar')
    plt.title(f'Target distribution: {target}')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outdir / f'target_distribution.png', dpi=150)
    plt.close()


def cramers_v_corr(x: pd.Series, y: pd.Series) -> float:
    contingency = pd.crosstab(x, y)
    if contingency.empty:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(contingency, correction=False)
    n = contingency.sum().sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = contingency.shape
    # Bias correction
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(1, (n - 1)))
    rcorr = r - (r - 1) ** 2 / max(1, (n - 1))
    kcorr = k - (k - 1) ** 2 / max(1, (n - 1))
    denom = max(1e-12, min((kcorr - 1), (rcorr - 1)))
    return np.sqrt(phi2corr / denom)


def point_biserial_corr(num: pd.Series, y: pd.Series) -> float:
    num = pd.to_numeric(num, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    mask = num.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    r, _ = stats.pointbiserialr(y[mask], num[mask])
    return r


def categorical_value_counts(df: pd.DataFrame, cat_cols: List[str], outdir: Path, top_n: int = 10):
    frames = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=False)
        head = vc.head(top_n).reset_index()
        head.columns = [c, 'count']
        head['column'] = c
        frames.append(head[['column', c, 'count']])
        plt.figure()
        vc.head(top_n).plot(kind='bar')
        plt.title(f'Value counts: {c} (top {top_n})')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(outdir / f'value_counts__{c}.png', dpi=150)
        plt.close()
    if len(frames) > 0:
        big = pd.concat(frames, ignore_index=True)
        big.to_csv(outdir / 'categorical_value_counts_topN.csv', index=False)


def bivariate_with_target(df: pd.DataFrame, target: str, cat_cols: List[str], num_cols: List[str], outdir: Path):
    if target not in df.columns:
        return

    y = pd.to_numeric(df[target], errors='coerce')

    # Categorical vs target
    rows_cat = []
    for c in cat_cols:
        # skip very high-cardinality categorical
        if df[c].nunique(dropna=False) > 20:
            continue
        tab = pd.crosstab(df[c], y)
        if tab.empty:
            continue
        if 1 in tab.columns and 0 in tab.columns:
            surv_rate = tab[1] / (tab[1] + tab[0])
        else:
            surv_rate = tab.div(tab.sum(axis=1), axis=0).iloc[:, -1]
        rows_cat.append({'column': c, 'cramers_v': cramers_v_corr(df[c], y)})

        plt.figure()
        surv_rate.sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Survival rate by {c}')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(outdir / f'survival_rate__by__{c}.png', dpi=150)
        plt.close()
    if rows_cat:
        pd.DataFrame(rows_cat).to_csv(outdir / 'cat_vs_target__cramers_v.csv', index=False)

    # Numeric vs target
    rows_num = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors='coerce')
        g0 = x[y == 0].dropna()
        g1 = x[y == 1].dropna()
        if len(g0) >= 2 and len(g1) >= 2:
            try:
                u_stat, p_val = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            except Exception:
                u_stat, p_val = (np.nan, np.nan)
        else:
            u_stat, p_val = (np.nan, np.nan)
        rows_num.append({
            'column': c,
            'point_biserial_r': point_biserial_corr(x, y),
            'mannwhitney_u': u_stat,
            'mannwhitney_p': p_val,
            'median_y0': float(np.median(g0)) if len(g0) else np.nan,
            'median_y1': float(np.median(g1)) if len(g1) else np.nan,
        })

        plt.figure()
        plt.boxplot([g0, g1], labels=['y=0', 'y=1'], showmeans=True)
        plt.title(f'{c} by target')
        plt.tight_layout()
        plt.savefig(outdir / f'num__{c}__by_target_boxplot.png', dpi=150)
        plt.close()
    if rows_num:
        pd.DataFrame(rows_num).to_csv(outdir / 'num_vs_target__stats.csv', index=False)


# ---------------------------
# Feature Engineering
# ---------------------------

def feature_engineering_preview(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Family size and "is_alone"
    sib_col = None
    if 'sibsp' in df.columns:
        sib_col = 'sibsp'
    elif 'sib_sp' in df.columns:
        sib_col = 'sib_sp'
    if sib_col and 'parch' in df.columns:
        sib = pd.to_numeric(df[sib_col], errors='coerce').fillna(0)
        par = pd.to_numeric(df['parch'], errors='coerce').fillna(0)
        df['family_size'] = sib + par + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)
        if 'fare' in df.columns:
            fare = pd.to_numeric(df['fare'], errors='coerce')
            denom = df['family_size'].replace({0: np.nan})
            df['fare_per_person'] = fare / denom

    # Age bins
    if 'age' in df.columns:
        age = pd.to_numeric(df['age'], errors='coerce')
        df['age_bin'] = pd.cut(
            age, bins=[-np.inf, 12, 18, 40, 60, np.inf],
            labels=['0-12', '13-18', '19-40', '41-60', '60+']
        )

    # Title from name
    if 'name' in df.columns:
        df['title'] = df['name'].astype(str).str.extract(r',\s*([^.]+)\.')[0].str.strip()

    # Deck from cabin
    if 'cabin' in df.columns:
        deck = df['cabin'].astype(str).str[0]
        df['deck'] = deck.replace({'n': np.nan, 'N': np.nan})

    # Ticket prefix
    if 'ticket' in df.columns:
        tp = df['ticket'].astype(str).str.replace(r'\d', '', regex=True).str.replace('.', '', regex=False).str.strip()
        df['ticket_prefix'] = tp.replace('', np.nan)

    return df


# ---------------------------
# MI screening & Baseline
# ---------------------------

def mutual_info_screening(df: pd.DataFrame, target: str, outdir: Path):
    """
    Computes mutual information on one-hot encoded features (dense to avoid
    'Sparse matrix X can't have continuous features').
    """
    # Lazy imports to keep script light without sklearn
    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import mutual_info_classif
    except Exception:
        return
    if target not in df.columns:
        return

    # Prepare
    df = df.dropna(subset=[target]).copy()
    y = pd.to_numeric(df[target], errors='coerce').astype(int)

    cat_cols, num_cols = infer_feature_types(df, target)
    if not cat_cols and not num_cols:
        return
    X = df[cat_cols + num_cols]

    # OHE -> dense for compatibility across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sklearn >=1.2
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                               ('oh', ohe)])

    pre = ColumnTransformer([('num', num_pipe, num_cols),
                             ('cat', cat_pipe, cat_cols)])

    X_enc = pre.fit_transform(X)

    # Just in case, ensure dense
    try:
        import scipy.sparse as sp
        if sp.issparse(X_enc):
            X_enc = X_enc.toarray()
    except Exception:
        pass

    mi = mutual_info_classif(X_enc, y, discrete_features='auto', random_state=42)
    pd.Series(mi).sort_values(ascending=False).to_csv(outdir / 'mutual_info_scores.csv', header=['mi'])


def quick_baseline(df: pd.DataFrame, target: str, outdir: Path):
    """
    Simple Logistic Regression baseline with dense OHE.
    Saves metrics, confusion matrix and ROC curve.
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            roc_curve, confusion_matrix, classification_report
        )
    except Exception:
        return
    if target not in df.columns:
        return

    df = df.dropna(subset=[target]).copy()
    y = pd.to_numeric(df[target], errors='coerce').astype(int)

    cat_cols, num_cols = infer_feature_types(df, target)
    if not cat_cols and not num_cols:
        return
    X = df[cat_cols + num_cols]

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    num_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                               ('oh', ohe)])

    pre = ColumnTransformer([('num', num_pipe, num_cols),
                             ('cat', cat_pipe, cat_cols)])

    clf = LogisticRegression(max_iter=200, class_weight='balanced')
    pipe = Pipeline(steps=[('pre', pre), ('clf', clf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
    }
    save_json(metrics, outdir / 'baseline_metrics.json')
    save_text(classification_report(y_test, y_pred), outdir / 'baseline_classification_report.txt')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix (Baseline)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, str(z), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(outdir / 'baseline_confusion_matrix.png', dpi=150)
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (Baseline)')
    plt.tight_layout()
    plt.savefig(outdir / 'baseline_roc_curve.png', dpi=150)
    plt.close()


# ---------------------------
# Summary & Kaggle download
# ---------------------------

def write_summary_md(df: pd.DataFrame, target: str, outdir: Path):
    parts = []
    parts.append(f"# EDA Summary\n")
    parts.append(f"- Rows: {df.shape[0]}, Cols: {df.shape[1]}\n")
    if target in df.columns:
        vc = df[target].value_counts(dropna=False).to_dict()
        parts.append(f"- Target `{target}` distribution: {vc}\n")
    cat_cols, num_cols = infer_feature_types(df, target if target in df.columns else None)
    parts.append(f"- Numeric columns: {num_cols}\n")
    parts.append(f"- Categorical columns: {cat_cols}\n")
    parts.append("\nArtifacts are saved as CSV/PNG in this folder.\n")
    save_text("\n".join(parts), outdir / 'SUMMARY.md')


def kaggle_download_titanic(download_dir: Path) -> Path:
    """
    Download Titanic competition files using Kaggle API or CLI. Returns path to train.csv.
    Requires:
      - ~/.kaggle/kaggle.json with proper permissions (chmod 600), and
      - you have joined the competition.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    zip_path = download_dir / "titanic.zip"

    # Try Python API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("titanic", path=str(download_dir), quiet=False)
    except Exception:
        # Fallback to CLI
        cmd = ["kaggle", "competitions", "download", "-c", "titanic", "-p", str(download_dir)]
        subprocess.check_call(cmd)

    # Unzip
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(download_dir)

    train_csv = download_dir / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found in {download_dir}.")
    return train_csv


# ---------------------------
# Orchestrator & CLI
# ---------------------------

def run_full_eda(df: pd.DataFrame, target: str, outdir: Path, run_baseline: bool = False):
    ensure_output_dir(outdir)
    df = normalize_columns(df)

    # Overview & missingness
    basic_overview(df, outdir)
    plot_missingness(df, outdir)

    # Target analysis
    if target in df.columns:
        target_analysis(df, target, outdir)

    # Feature engineering preview
    df_fe = feature_engineering_preview(df)

    # Types & univariate
    cat_cols, num_cols = infer_feature_types(df_fe, target if target in df_fe.columns else None)
    categorical_value_counts(df_fe, cat_cols, outdir, top_n=10)

    # Bivariate & MI
    if target in df_fe.columns:
        bivariate_with_target(df_fe, target, cat_cols, num_cols, outdir)
        mutual_info_screening(df_fe, target, outdir)
        if run_baseline:
            quick_baseline(df_fe, target, outdir)

    # Summary
    write_summary_md(df_fe, target, outdir)
    return df_fe


def main():
    parser = argparse.ArgumentParser(description='Titanic-style EDA script (enhanced, final)')
    parser.add_argument('--input', help='Path to input CSV')
    parser.add_argument('--target', default='survived', help='Target column name (default: survived)')
    parser.add_argument('--output', default='eda_report', help='Output directory')
    parser.add_argument('--baseline', action='store_true', help='Run quick baseline model')
    parser.add_argument('--download-kaggle', action='action', nargs='?', const=True, default=False,
                        help='Download Titanic from Kaggle and use train.csv')
    parser.add_argument('--kaggle-dir', default='kaggle_data', help='Directory to store Kaggle files')
    args = parser.parse_args()

    outdir = Path(args.output)

    if args.download_kaggle:
        train_path = kaggle_download_titanic(Path(args.kaggle_dir))
        df = pd.read_csv(train_path)
    else:
        if not args.input:
            print("Either --input must be provided or use --download-kaggle.", flush=True)
            raise SystemExit(2)
        df = pd.read_csv(args.input)

    run_full_eda(df, args.target, outdir, run_baseline=args.baseline)
    print(f"EDA complete. Outputs saved to: {outdir.resolve()}")


if __name__ == '__main__':
    main()
