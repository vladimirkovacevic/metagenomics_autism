"""Clinical modality analysis (PLAN_MM.md §4.3).

Univariate group comparisons (autism vs control) on the broad clinical set
(continuous via Mann-Whitney; categorical via chi-square/Fisher), BH-FDR;
age & sex distribution figures; clustering heatmap of the summary scores.
Figures + tables -> results/clinical/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import mm_common as C

log = C.get_logger("clinical")
OUT = os.path.join(C.RESULTS, "clinical")
C.set_style()
PALETTE = C.GROUP_COLORS


def main():
    C.ensure_dirs(OUT)
    analysis = pd.read_csv(os.path.join(C.PRE, "clinical_analysis.tsv"), sep="\t", index_col=0)
    types = pd.read_csv(os.path.join(C.PRE, "clinical_types.tsv"), sep="\t", index_col=0)["type"]
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)
    group = analysis["group"].astype(str)
    a_mask, k_mask = group == "autism", group == "control"
    log.info("clinical analysis: %d samples, %d variables tested", len(analysis), len(types))

    # ---- univariate group comparisons ---- #
    rows = {}
    for var, vtype in types.items():
        if var not in analysis.columns:
            continue
        col = analysis[var]
        if vtype == "continuous":
            num = pd.to_numeric(col, errors="coerce")
            a, k = num[a_mask].dropna(), num[k_mask].dropna()
            if len(a) < 5 or len(k) < 5:
                continue
            try:
                p = stats.mannwhitneyu(a, k, alternative="two-sided")[1]
            except ValueError:
                continue
            rows[var] = {"type": "continuous", "test": "Mann-Whitney",
                         "autism_median": a.median(), "control_median": k.median(),
                         "effect": a.median() - k.median(), "pval": p, "n": len(a) + len(k)}
        else:
            cat = col.astype("category")
            ct = pd.crosstab(cat, group)
            if ct.shape[0] < 2 or ct.values.sum() < 20:
                continue
            try:
                if ct.shape == (2, 2):
                    p = stats.fisher_exact(ct.values)[1]; test = "Fisher"
                else:
                    p = stats.chi2_contingency(ct.values)[1]; test = "Chi2"
            except ValueError:
                continue
            rows[var] = {"type": "categorical", "test": test,
                         "autism_median": np.nan, "control_median": np.nan,
                         "effect": np.nan, "pval": p, "n": int(ct.values.sum())}

    res = pd.DataFrame(rows).T
    res["pval"] = pd.to_numeric(res["pval"])
    res["qval"] = multipletests(res["pval"], method="fdr_bh")[1]
    res = res.sort_values("qval")
    res.to_csv(os.path.join(OUT, "clinical_group_tests.csv"))
    sig = res[res["qval"] < 0.05]
    log.info("clinical: %d/%d variables significant at q<0.05", len(sig), len(res))
    log.info("top hits: %s", list(sig.head(8).index))

    # ---- age & sex distribution ---- #
    age = pd.to_numeric(cov["age"], errors="coerce")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for g in ["autism", "control"]:
        ax1.hist(age[group == g].dropna(), bins=np.arange(0, 22, 2), alpha=0.6,
                 color=PALETTE[g], label=g)
    ax1.set_xlabel("age (2024 − birth year)"); ax1.set_ylabel("samples")
    ax1.set_title("Age distribution by group"); ax1.legend()
    sexct = pd.crosstab(cov["sex"], group)
    sexct.plot(kind="bar", ax=ax2, color=[PALETTE["autism"], PALETTE["control"]])
    ax2.set_title("Sex distribution by group"); ax2.set_xlabel("sex"); ax2.set_ylabel("samples")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "age_sex_distribution.png"), dpi=130); plt.close(fig)

    # ---- clinical missingness distribution (PLAN_MM.md §6.3) ---- #
    miss = analysis.drop(columns=["group"], errors="ignore").isna().mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(miss.values, bins=30, color=C.CAT_PALETTE[0])
    ax.set_xlabel("fraction missing"); ax.set_ylabel("number of variables")
    ax.set_title(f"Clinical variable missingness ({len(miss)} variables)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "missingness.png"), dpi=130); plt.close(fig)
    log.info("missingness: median %.0f%% across %d clinical variables",
             100 * miss.median(), len(miss))

    # ---- clustering heatmap of summary scores (z-scored, complete cases) ---- #
    score_cols = [c for c in C.CLINICAL_SCORE_BLOCK if c in analysis.columns]
    scores = analysis[score_cols].apply(pd.to_numeric, errors="coerce")
    z = ((scores - scores.mean()) / scores.std()).dropna(how="any")
    if len(z) > 5:
        row_order = leaves_list(linkage(z.values, method="ward"))
        col_order = leaves_list(linkage(z.T.values, method="ward"))
        zz = z.iloc[row_order, col_order]
        fig, ax = plt.subplots(figsize=(10, 9))
        im = ax.imshow(zz.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(zz.shape[1]))
        ax.set_xticklabels([c[:30] for c in zz.columns], rotation=90, fontsize=6)
        ax.set_yticks([])
        rc = [PALETTE[group.loc[s]] for s in zz.index]
        ax.scatter([-1.2] * len(zz), range(len(zz)), c=rc, s=8, clip_on=False, marker="s")
        ax.set_title("Clinical summary scores (z), ward-clustered; left bar = group")
        fig.colorbar(im, ax=ax, shrink=0.5, label="z-score")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "scores_heatmap.png"), dpi=130); plt.close(fig)
        log.info("score heatmap: %d complete-case samples x %d scores", *zz.shape)

    # ---- L1-regularized logistic regression: clinical -> group ---------- #
    # Use only variables available in BOTH groups (>=50% non-missing each) to
    # avoid leakage from autism-only questionnaire scores.
    y = (group == "autism").astype(int)
    feat = analysis.drop(columns=["group"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    ok = [c for c in feat.columns
          if feat.loc[a_mask, c].notna().mean() >= 0.5 and feat.loc[k_mask, c].notna().mean() >= 0.5
          and feat[c].nunique() > 1]
    X = feat[ok]
    log.info("logistic regression: %d clinical features available in both groups", len(ok))
    pipe = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=1)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"AUC": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
               "F1": f1_score(y, pred), "n_features": len(ok)}
    log.info("logistic CV: AUC=%.3f acc=%.3f F1=%.3f", metrics["AUC"], metrics["accuracy"], metrics["F1"])
    pipe.fit(X, y)
    coefs = pd.Series(pipe.named_steps["logisticregression"].coef_[0], index=ok)
    coefs = coefs[coefs != 0].sort_values()
    coefs.to_csv(os.path.join(OUT, "logreg_coefficients.csv"))
    pd.Series(metrics).to_csv(os.path.join(OUT, "logreg_metrics.csv"))
    log.info("logistic regression retained %d non-zero features", len(coefs))

    if len(coefs):
        top = pd.concat([coefs.head(10), coefs.tail(10)]).drop_duplicates()
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
        ax.barh(range(len(top)), top.values,
                color=np.where(top.values > 0, C.SIG_UP, C.SIG_DOWN))
        ax.set_yticks(range(len(top))); ax.set_yticklabels([c[:34] for c in top.index], fontsize=7)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel("L1 logistic coefficient (+ → autism)")
        ax.set_title(f"Clinical predictors of group\n(L1 logistic; CV AUC={metrics['AUC']:.2f}, "
                     f"acc={metrics['accuracy']:.2f})")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "logreg_coefficients.png"), dpi=160); plt.close(fig)

    log.info("clinical analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
