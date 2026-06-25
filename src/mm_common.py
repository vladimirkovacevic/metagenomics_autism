"""Shared utilities for the multi-omics analysis pipeline (PLAN_MM.md).

Centralizes paths, file logging, data loading, clinical variable selection,
and the CLR transform so every stage uses identical conventions.
"""
from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Visual identity (PLAN_MM.md §6 color scheme) + Nature-style figure defaults
# --------------------------------------------------------------------------- #
GROUP_COLORS = {"autism": "#3287be", "control": "#ff7d55"}      # primary group palette
CAT_PALETTE = ["#3287be", "#f0b4dc", "#b9e187", "#ff7d55"]       # categorical palette
SIG_UP = "#d7191c"     # upregulated / enriched in autism
SIG_DOWN = "#2c7bb6"   # downregulated / enriched in control
SIG_NS = "#B0B0B0"     # not significant


def set_style():
    """Apply a clean Nature-journal-like matplotlib style."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "font.family": "sans-serif",
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def sig_color(coef, q, alpha=0.05):
    """Color a feature by significance and effect direction (autism − control)."""
    if q >= alpha or not np.isfinite(q):
        return SIG_NS
    return SIG_UP if coef > 0 else SIG_DOWN

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

METAGENOMICS_TSV = os.path.join(DATA, "metagenomics.tsv")
METABOLOMICS_TSV = os.path.join(DATA, "metabolomics.tsv")
CLINICAL_TSV = os.path.join(DATA, "clinical.tsv")

PRE = os.path.join(RESULTS, "preprocessed")
LOGS = os.path.join(RESULTS, "logs")

# Reference year for deriving age from birth year (sample-collection year; see PLAN_MM.md §0).
AGE_REFERENCE_YEAR = 2024
BIRTH_YEAR_COL = "godina_rodjenja_deteta"
SEX_COL = "pol_deteta_ispitanika"

# Clinical questionnaire variable blocks (1-based positions in clinical.sav; PLAN_MM.md §0).
# Raw item blocks are EXCLUDED; only the calculated summary scores are kept.
SSDS_ITEMS = (138, 208)   # excluded
SDQ_ITEMS = (209, 241)    # excluded
GIRBI_ITEMS = (242, 353)  # kept (many items not captured by factor scores)

SSDS_SCORES = [
    "SSDS_Social_Motivation", "SSDS_Social_Affiliation",
    "SSDS_Expressive_Social_Communication", "SSDS_Social_Recognition",
    "SSDS_Unusual_Approach",
]
SDQ_SCORES = [
    "SDQ_Emotional_Difficulties", "SDQ_Conduct_Difficulties", "SDQ_Hyperactivity",
    "SDQ_Peer_Difficulties", "SDQ_Prosocial_Scale", "SDQ_Total_Difficulties",
    "SDQ_Internalising_Difficulties", "SDQ_Externalising_Difficulties",
    "SDQ_Parent_reported_impact",
]
GIRBI_FACTORS = [
    "GIRBI_Bowel_Movement_Pain_Factor_1", "GIRBI_Aggressive_Disruptive_At_Mealtimes_Factor_2",
    "GIRBI_Particular_With_Food_Factor_3", "GIRBI_Abdominal_Pain_Upset_Stomach_Factor_4",
    "GIRBI_Refuses_Food_Factor_5", "GIRBI_Constipation_Encopresis_Factor_6",
    "GIRBI_Motor_Other_Behaviors_Factor_7", "GIRBI_Total",
]
# Clean continuous summary scores used as the clinical INTEGRATION block (PLAN_MM.md §5).
CLINICAL_SCORE_BLOCK = SSDS_SCORES + SDQ_SCORES + GIRBI_FACTORS


# --------------------------------------------------------------------------- #
# Logging — console + per-stage file (success criterion: logs saved to file)
# --------------------------------------------------------------------------- #
def get_logger(stage: str) -> logging.Logger:
    os.makedirs(LOGS, exist_ok=True)
    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(LOGS, f"{stage}.log"), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_harmonized():
    """Load the three harmonized matrices (samples as rows, identical order)."""
    mg = pd.read_csv(METAGENOMICS_TSV, sep="\t", index_col=0)
    mb = pd.read_csv(METABOLOMICS_TSV, sep="\t", index_col=0)
    cl = pd.read_csv(CLINICAL_TSV, sep="\t", index_col=0)
    assert list(mg.index) == list(mb.index) == list(cl.index), "harmonized order mismatch"
    return mg, mb, cl


def derive_age(clinical: pd.DataFrame) -> pd.Series:
    """age = AGE_REFERENCE_YEAR - birth_year (blanks/unparseable -> NaN)."""
    by = pd.to_numeric(clinical[BIRTH_YEAR_COL], errors="coerce")
    return (AGE_REFERENCE_YEAR - by).rename("age")


def get_group(clinical: pd.DataFrame) -> pd.Series:
    return clinical["group"].astype(str).rename("group")


def get_sex(clinical: pd.DataFrame) -> pd.Series:
    """Sex as a clean categorical (1=male, 2=female -> 'M'/'F'); other -> NaN."""
    s = pd.to_numeric(clinical[SEX_COL], errors="coerce")
    return s.map({1: "M", 2: "F"}).rename("sex")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def clr_transform(df: pd.DataFrame, pseudo: str = "multiplicative") -> pd.DataFrame:
    """Centered log-ratio transform of a samples x features compositional matrix.

    Uses a multiplicative replacement pseudocount (half the smallest non-zero
    value per matrix) to handle zeros, then CLR = log(x) - mean(log(x)) per row.
    """
    X = df.to_numpy(dtype=float)
    nz_min = X[X > 0].min()
    X = np.where(X == 0, nz_min / 2.0, X)
    X = X / X.sum(axis=1, keepdims=True)          # re-close to relative abundance
    logX = np.log(X)
    clr = logX - logX.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=df.index, columns=df.columns)


def build_design(cov: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix from covariates: group01 (autism=1), age, sex01 (M=1)."""
    d = pd.DataFrame(index=cov.index)
    d["group01"] = (cov["group"].astype(str) == "autism").astype(float)
    d["age"] = pd.to_numeric(cov["age"], errors="coerce")
    d["sex01"] = cov["sex"].map({"M": 1.0, "F": 0.0})
    return d


def differential_lm(features: pd.DataFrame, cov: pd.DataFrame, adjust=("age", "sex01")):
    """Per-feature OLS: feature ~ group01 + age + sex01, BH-FDR on the group term.

    Used for both metagenomics (CLR) and functional pathways (CLR). This is the
    all-Python substitute for ANCOM-BC/MaAsLin2 (see PLAN_MM.md D1 / logs).
    Returns a DataFrame indexed by feature with group coefficient, p, q, and
    per-group means; features/samples with insufficient data are skipped.
    """
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    d = build_design(cov)
    terms = ["group01"] + [a for a in adjust if a in d.columns]
    keep = d[terms].dropna().index
    d = d.loc[keep]
    X = sm.add_constant(d[terms])
    grp = d["group01"].astype(bool)

    rows = {}
    for feat in features.columns:
        y = pd.to_numeric(features.loc[keep, feat], errors="coerce")
        ok = y.notna()
        if ok.sum() < 10 or y[ok].nunique() < 3:
            continue
        try:
            res = sm.OLS(y[ok], X.loc[ok]).fit()
        except Exception:
            continue
        rows[feat] = {
            "coef_group": res.params.get("group01", np.nan),
            "pval": res.pvalues.get("group01", np.nan),
            "mean_autism": y[ok & grp].mean(),
            "mean_control": y[ok & ~grp].mean(),
            "n": int(ok.sum()),
        }
    out = pd.DataFrame(rows).T
    if out.empty:
        return out
    out["qval"] = multipletests(out["pval"], method="fdr_bh")[1]
    return out.sort_values("qval")


def classify_clinical_columns(clinical: pd.DataFrame):
    """Split clinical variables into continuous / categorical / dropped (free-text).

    Rules (PLAN_MM.md §0/§3): keep only numeric or low-cardinality categorical
    variables; exclude raw SSDS (138-208) and SDQ (209-241) item blocks, the ID
    column, the raw birth-year (replaced by derived age), and free-text columns.
    Returns (continuous_cols, categorical_cols, dropped_reasons).
    """
    cols = list(clinical.columns)
    excluded_positions = set(range(SSDS_ITEMS[0] - 1, SSDS_ITEMS[1])) | \
        set(range(SDQ_ITEMS[0] - 1, SDQ_ITEMS[1]))

    continuous, categorical, dropped = [], [], {}
    for i, c in enumerate(cols):
        if c in ("group", "sex", "age", "_code_norm", "Šifra", BIRTH_YEAR_COL):
            continue
        if i in excluded_positions:
            dropped[c] = "raw SSDS/SDQ item (excluded by plan)"
            continue
        col = clinical[c]
        num = pd.to_numeric(col, errors="coerce")
        n_nonnull = col.notna().sum()
        if n_nonnull == 0:
            dropped[c] = "all missing"
            continue
        # numeric if most non-null values parse as numbers
        if num.notna().sum() >= 0.8 * n_nonnull:
            nuniq = num.dropna().nunique()
            if nuniq <= 1:
                dropped[c] = "constant"
            elif nuniq <= 6:
                categorical.append(c)
            else:
                continuous.append(c)
        else:
            # string column: keep only if low-cardinality (true categorical)
            nuniq = col.dropna().astype(str).str.strip().nunique()
            if nuniq <= 6:
                categorical.append(c)
            else:
                dropped[c] = f"free-text / high-cardinality string ({nuniq} levels)"
    return continuous, categorical, dropped
