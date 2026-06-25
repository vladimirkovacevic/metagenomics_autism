"""MOFA2 unsupervised multi-omics factor analysis (PLAN_MM.md §5.1).

Two-block cross-group model (metagenomics CLR + functional pathways CLR) on all
181 samples. Trains with mofapy2, then reads the HDF5 model to produce:
  - variance decomposition (variance explained per factor per view)
  - factor-covariate associations (factor vs group/age/sex)
  - top feature weights per factor per view
  - factor scatter plots coloured by group
Outputs -> results/integration/mofa/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import mm_common as C

log = C.get_logger("mofa")
OUT = os.path.join(C.RESULTS, "integration", "mofa")
C.set_style()
PALETTE = C.GROUP_COLORS
N_FACTORS = 8
TOP_VAR_FEATURES = 500  # cap per view to the most variable features for stability


def most_variable(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[df.var().sort_values(ascending=False).head(n).index]


def main():
    C.ensure_dirs(OUT)
    mg = pd.read_csv(os.path.join(C.PRE, "metagenomics_clr.tsv"), sep="\t", index_col=0)
    mb = pd.read_csv(os.path.join(C.PRE, "metabolomics_clr.tsv"), sep="\t", index_col=0)
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)

    mg = most_variable(mg, TOP_VAR_FEATURES)
    mb = most_variable(mb, min(TOP_VAR_FEATURES, mb.shape[1]))
    log.info("MOFA blocks: metagenomics %s, pathways %s", mg.shape, mb.shape)

    from mofapy2.run.entry_point import entry_point
    samples = list(mg.index)
    # mofapy2 expects per-view list of arrays (groups x views); single group here.
    data = [[mg.loc[samples].values], [mb.loc[samples].values]]

    ent = entry_point()
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data, views_names=["metagenomics", "pathways"],
                        samples_names=[samples],
                        features_names=[list(mg.columns), list(mb.columns)])
    ent.set_model_options(factors=N_FACTORS, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(iter=1000, convergence_mode="fast", seed=1, verbose=False)
    ent.build()
    ent.run()
    model_path = os.path.join(OUT, "mofa_model.hdf5")
    ent.save(model_path, save_data=False)
    log.info("MOFA model trained and saved -> %s", os.path.relpath(model_path, C.ROOT))

    # ---- read model back via h5py ---- #
    import h5py
    with h5py.File(model_path, "r") as f:
        views = [v.decode() if isinstance(v, bytes) else v for v in f["views"]["views"][()]]
        Z = f["expectations"]["Z"]["group0"][()].T          # samples x factors
        Wd = {v: f["expectations"]["W"][v][()] for v in views}  # factors x features
        r2 = f["variance_explained"]["r2_per_factor"]["group0"][()]  # views x factors
    factors = pd.DataFrame(Z, index=samples,
                           columns=[f"Factor{i+1}" for i in range(Z.shape[1])])
    factors.join(cov["group"]).to_csv(os.path.join(OUT, "factors.csv"))

    # ---- variance decomposition ---- #
    r2 = np.atleast_2d(r2)
    if r2.shape[0] != len(views):
        r2 = r2.T
    r2df = pd.DataFrame(r2, index=views, columns=factors.columns)
    r2df.to_csv(os.path.join(OUT, "variance_explained.csv"))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(r2df.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(views))); ax.set_yticklabels(views)
    ax.set_xticks(range(r2df.shape[1])); ax.set_xticklabels(r2df.columns, rotation=45, ha="right")
    for (i, j), v in np.ndenumerate(r2df.values):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="w" if v < r2df.values.max() * 0.6 else "k", fontsize=7)
    ax.set_title("MOFA variance explained (%) per factor per view")
    fig.colorbar(im, ax=ax, shrink=0.7, label="% variance")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "variance_explained.png"), dpi=130); plt.close(fig)
    log.info("variance explained per view: %s", r2df.sum(axis=1).round(1).to_dict())

    # ---- factor-covariate association ---- #
    age = pd.to_numeric(cov["age"], errors="coerce")
    grp01 = (cov["group"].astype(str) == "autism").astype(float)
    sex01 = cov["sex"].map({"M": 1.0, "F": 0.0})
    assoc = {}
    for fac in factors.columns:
        z = factors[fac]
        a = z[grp01 == 1]; k = z[grp01 == 0]
        p_grp = stats.mannwhitneyu(a, k, alternative="two-sided")[1]
        ok = age.notna()
        r_age, p_age = stats.spearmanr(z[ok], age[ok])
        oks = sex01.notna()
        p_sex = stats.mannwhitneyu(z[(sex01 == 1)], z[(sex01 == 0)], alternative="two-sided")[1]
        assoc[fac] = {"p_group": p_grp, "r_age": r_age, "p_age": p_age, "p_sex": p_sex,
                      "delta_group": a.mean() - k.mean()}
    assoc = pd.DataFrame(assoc).T
    assoc.to_csv(os.path.join(OUT, "factor_covariate_association.csv"))
    log.info("factors associated with group (p<0.05): %s",
             list(assoc.index[assoc["p_group"] < 0.05]))

    # scatter of the two group-most-associated factors
    best = assoc["p_group"].astype(float).sort_values().index[:2].tolist()
    if len(best) < 2:
        best = ["Factor1", "Factor2"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for g in ["autism", "control"]:
        m = cov["group"].astype(str).values == g
        ax.scatter(factors[best[0]].values[m], factors[best[1]].values[m],
                   s=28, alpha=0.7, c=PALETTE[g], label=g)
    ax.set_xlabel(f"{best[0]} (group p={assoc.loc[best[0],'p_group']:.1e})")
    ax.set_ylabel(f"{best[1]} (group p={assoc.loc[best[1],'p_group']:.1e})")
    ax.set_title("MOFA factors most associated with group"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "factor_scatter.png"), dpi=130); plt.close(fig)

    # ---- top feature weights for the top group factor ---- #
    topf = best[0]
    fi = factors.columns.get_loc(topf)
    for v in views:
        cols = mg.columns if v == "metagenomics" else mb.columns
        w = pd.Series(Wd[v][fi], index=cols).sort_values()
        sel = pd.concat([w.head(10), w.tail(10)])
        sel.to_csv(os.path.join(OUT, f"top_weights_{topf}_{v}.csv"))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(sel)), sel.values,
                color=np.where(sel.values > 0, C.SIG_UP, C.SIG_DOWN))
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([c.split(":")[0][:42] for c in sel.index], fontsize=7)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_title(f"{topf} top weights — {v}")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"weights_{topf}_{v}.png"), dpi=130); plt.close(fig)

    # ---- factor vs clinical-variable correlation heatmap ---- #
    analysis = pd.read_csv(os.path.join(C.PRE, "clinical_analysis.tsv"), sep="\t", index_col=0)
    scores = pd.read_csv(os.path.join(C.PRE, "clinical_scores.tsv"), sep="\t", index_col=0)
    clin_vars = pd.DataFrame(index=factors.index)
    clin_vars["age"] = age
    clin_vars["sex"] = sex01
    clin_vars["group(ASD)"] = grp01
    for c in C.CLINICAL_SCORE_BLOCK:
        if c in scores.columns:
            clin_vars[c] = pd.to_numeric(scores[c], errors="coerce")
    cmat = pd.DataFrame(index=factors.columns, columns=clin_vars.columns, dtype=float)
    for f in factors.columns:
        for v in clin_vars.columns:
            d = pd.concat([factors[f], clin_vars[v]], axis=1).dropna()
            cmat.loc[f, v] = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0] if len(d) > 5 else np.nan
    cmat.astype(float).to_csv(os.path.join(OUT, "factor_clinical_correlation.csv"))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    im = ax.imshow(cmat.astype(float).values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_yticks(range(len(cmat))); ax.set_yticklabels(cmat.index)
    ax.set_xticks(range(cmat.shape[1])); ax.set_xticklabels([c[:24] for c in cmat.columns],
                                                            rotation=90, fontsize=6)
    ax.set_title("MOFA factor – clinical variable correlation (Spearman; scores autism-only)")
    fig.colorbar(im, ax=ax, shrink=0.7, label="rho")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "factor_clinical_heatmap.png"), dpi=150); plt.close(fig)
    log.info("factor-clinical correlation heatmap written")

    # ---- XGBoost on factors -> group (stratified 5-fold CV) ---- #
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
    from xgboost import XGBClassifier
    y = grp01.astype(int).values
    Xf = factors.values
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                        subsample=0.9, eval_metric="logloss", random_state=1)
    cvf = StratifiedKFold(5, shuffle=True, random_state=1)
    proba = cross_val_predict(clf, Xf, y, cv=cvf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    xgb_metrics = {"AUC": roc_auc_score(y, proba), "accuracy": accuracy_score(y, pred),
                   "F1": f1_score(y, pred)}
    pd.Series(xgb_metrics).to_csv(os.path.join(OUT, "xgboost_factor_metrics.csv"))
    log.info("XGBoost on MOFA factors -> group (5-fold CV): AUC=%.3f acc=%.3f F1=%.3f",
             xgb_metrics["AUC"], xgb_metrics["accuracy"], xgb_metrics["F1"])
    clf.fit(Xf, y)
    imp = pd.Series(clf.feature_importances_, index=factors.columns).sort_values()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.barh(range(len(imp)), imp.values, color=C.CAT_PALETTE[0])
    ax1.set_yticks(range(len(imp))); ax1.set_yticklabels(imp.index)
    ax1.set_xlabel("XGBoost importance"); ax1.set_title("Factor importance")
    ax2.bar(list(xgb_metrics), list(xgb_metrics.values()), color=C.CAT_PALETTE[3])
    ax2.set_ylim(0, 1); ax2.set_title("XGBoost 5-fold CV (factors → group)")
    for i, v in enumerate(xgb_metrics.values()):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "xgboost_factors.png"), dpi=150); plt.close(fig)

    log.info("MOFA analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
