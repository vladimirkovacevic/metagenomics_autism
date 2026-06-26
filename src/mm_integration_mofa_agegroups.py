"""Age-group-specific MOFA2 integration (PLAN_MM.md §5.1, age-stratified).

Trains a separate MOFA2 model within each developmental age group (0-7, 8-12,
13+) on the two CLR omics blocks, then tests every latent factor for association
with diagnosis (Mann-Whitney, BH-FDR across factors). Reports, per age group:
total variance explained per view, and the most group-associated factor with its
FDR-adjusted p-value (statistical significance). Outputs -> results/integration/mofa/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import mm_common as C

C.set_style()
log = C.get_logger("mofa_agegroups")
OUT = os.path.join(C.RESULTS, "integration", "mofa")
PALETTE = C.GROUP_COLORS
N_FACTORS = 5          # fewer factors than the whole-cohort model (smaller N per stratum)
TOP_VAR = 300          # cap features per view for stability


def most_variable(df, n):
    return df[df.var().sort_values(ascending=False).head(min(n, df.shape[1])).index]


def train_mofa(mg, mb, samples):
    from mofapy2.run.entry_point import entry_point
    data = [[mg.loc[samples].values], [mb.loc[samples].values]]
    ent = entry_point()
    ent.set_data_options(scale_views=True)
    ent.set_data_matrix(data, views_names=["metagenomics", "pathways"],
                        samples_names=[list(samples)],
                        features_names=[list(mg.columns), list(mb.columns)])
    ent.set_model_options(factors=N_FACTORS, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(iter=1000, convergence_mode="fast", seed=1, verbose=False)
    ent.build(); ent.run()
    import tempfile, h5py
    fp = os.path.join(tempfile.gettempdir(), "mofa_agetmp.hdf5")
    ent.save(fp, save_data=False)
    with h5py.File(fp, "r") as f:
        views = [v.decode() if isinstance(v, bytes) else v for v in f["views"]["views"][()]]
        Z = f["expectations"]["Z"]["group0"][()].T
        r2 = np.atleast_2d(f["variance_explained"]["r2_per_factor"]["group0"][()])
    if r2.shape[0] != len(views):
        r2 = r2.T
    return Z, dict(zip(views, r2.sum(axis=1)))


def main():
    C.ensure_dirs(OUT)
    mg = pd.read_csv(os.path.join(C.PRE, "metagenomics_clr.tsv"), sep="\t", index_col=0)
    mb = pd.read_csv(os.path.join(C.PRE, "metabolomics_clr.tsv"), sep="\t", index_col=0)
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)
    ag = C.age_group(cov["age"]).astype(object)
    group = cov["group"].astype(str)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, agl in zip(axes, C.AGE_GROUP_LABELS):
        samples = [s for s in mg.index if ag.get(s) == agl]
        gsub = group.loc[samples]
        if gsub.nunique() < 2 or len(samples) < 12:
            ax.axis("off"); continue
        mgv = most_variable(mg.loc[samples], TOP_VAR)
        mbv = most_variable(mb.loc[samples], TOP_VAR)
        Z, var = train_mofa(mgv, mbv, samples)
        factors = pd.DataFrame(Z, index=samples,
                               columns=[f"F{i+1}" for i in range(Z.shape[1])])
        y = (gsub.values == "autism")
        pvals = [stats.mannwhitneyu(factors[c].values[y], factors[c].values[~y],
                                    alternative="two-sided").pvalue for c in factors.columns]
        qvals = multipletests(pvals, method="fdr_bh")[1]
        order = np.argsort(pvals)
        best, second = factors.columns[order[0]], factors.columns[order[min(1, len(order)-1)]]
        rows.append({"age_group": agl, "n": len(samples),
                     "best_factor": best, "p_best": pvals[order[0]], "q_best": qvals[order[0]],
                     "var_metagenomics": round(var.get("metagenomics", np.nan), 1),
                     "var_pathways": round(var.get("pathways", np.nan), 1)})
        for grp in ["autism", "control"]:
            m = gsub.values == grp
            ax.scatter(factors[best].values[m], factors[second].values[m],
                       s=26, alpha=0.75, c=PALETTE[grp], label=grp)
        ax.set_xlabel(f"{best}"); ax.set_ylabel(f"{second}")
        ax.set_title(f"{agl} (n={len(samples)})\nbest factor {best}: group q={qvals[order[0]]:.3f}",
                     fontsize=9)
        if ax is axes[0]:
            ax.legend(fontsize=8)
        log.info("MOFA %s: n=%d best=%s group p=%.3g q=%.3g var(mg/pw)=%.1f/%.1f", agl, len(samples),
                 best, pvals[order[0]], qvals[order[0]], var.get("metagenomics", np.nan),
                 var.get("pathways", np.nan))
    fig.suptitle("Age-group-specific MOFA2: samples on the two most group-associated factors")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mofa_agegroup.png"), dpi=140); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "mofa_agegroup_assoc.csv"), index=False)
    log.info("age-group MOFA complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
