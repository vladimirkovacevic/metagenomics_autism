"""Functional pathway (HUMAnN) modality analysis (PLAN_MM.md §4.2).

NOTE: `metabolomics.tsv` is HUMAnN *functional pathway* data, not true
metabolites; results are labelled accordingly. PCA ordination + age/sex-adjusted
differential pathway abundance. Figures + tables -> results/metabolomics/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

import mm_common as C

log = C.get_logger("metabolomics")
OUT = os.path.join(C.RESULTS, "metabolomics")
C.set_style()
PALETTE = C.GROUP_COLORS


def main():
    C.ensure_dirs(OUT)
    clr = pd.read_csv(os.path.join(C.PRE, "metabolomics_clr.tsv"), sep="\t", index_col=0)
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)
    group = cov["group"].astype(str)
    log.info("functional-pathway analysis: %d samples x %d pathways", *clr.shape)

    # ---- PCA ordination ---- #
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(clr.values)
    expl = pca.explained_variance_ratio_[:2] * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    for g in ["autism", "control"]:
        m = group.values == g
        ax.scatter(pcs[m, 0], pcs[m, 1], s=28, alpha=0.7, c=PALETTE[g], label=g)
    ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax.set_title("Functional pathways PCA (CLR) by group"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pca.png"), dpi=130); plt.close(fig)

    # ---- differential pathway abundance (CLR-LM, age/sex-adjusted, BH-FDR) ---- #
    da = C.differential_lm(clr, cov)
    da.to_csv(os.path.join(OUT, "differential_pathways.csv"))
    sig = da[da["qval"] < 0.05]
    log.info("differential pathways: %d significant at q<0.05 (of %d tested)", len(sig), len(da))

    fig, ax = plt.subplots(figsize=(7, 5))
    nlq = -np.log10(da["qval"].clip(lower=1e-300))
    colors = [C.sig_color(c, q) for c, q in zip(da["coef_group"], da["qval"])]
    ax.scatter(da["coef_group"], nlq, s=16, c=colors, alpha=0.8, edgecolors="none")
    ax.axhline(-np.log10(0.05), ls="--", c="k", lw=0.8)
    ax.set_xlabel("CLR coefficient (autism − control, age/sex-adj)")
    ax.set_ylabel("-log10(q)")
    ax.set_title(f"Differential pathways volcano ({len(sig)} sig at q<0.05)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "differential_volcano.png"), dpi=130); plt.close(fig)

    top = da.reindex(da["coef_group"].abs().sort_values(ascending=False).index).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(top)), top["coef_group"],
            color=np.where(top["coef_group"] > 0, C.SIG_UP, C.SIG_DOWN))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([t.split(":")[0][:45] for t in top.index], fontsize=7)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("CLR coefficient (autism − control)")
    ax.set_title("Top 20 pathways by effect size (red=↑autism, blue=↑control)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "differential_top_pathways.png"), dpi=130); plt.close(fig)

    # ---- group×age interaction (effect modification by age band) ---- #
    inter = C.interaction_lm(clr, cov)
    inter.to_csv(os.path.join(OUT, "differential_interaction.csv"))
    n_int = int((inter["q_interaction"] < 0.05).sum()) if not inter.empty else 0
    log.info("functional pathways: group×age interaction significant (q<0.05) in %d pathways", n_int)

    # ============ AGE-STRATIFIED HUMAnN ANALYSIS (per developmental group) ============ #
    AGES = C.AGE_GROUP_LABELS
    cov_ag = cov.assign(age_group=C.age_group(cov["age"]).astype(object))

    # per-stratum PERMANOVA + PCoA on functional CLR (statistical significance per group)
    brows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, agl in zip(axes, AGES):
        idx = list(cov_ag.index[cov_ag.age_group == agl]); gsub = group.loc[idx]
        if gsub.nunique() < 2 or len(idx) < 8:
            ax.axis("off"); continue
        dm = DistanceMatrix(squareform(pdist(clr.loc[idx].values)), ids=idx)
        pe = permanova(dm, grouping=list(gsub), permutations=999)
        brows.append({"age_group": agl, "n": len(idx),
                      "PERMANOVA_F": pe["test statistic"], "PERMANOVA_p": pe["p-value"]})
        ordi = pcoa(dm, number_of_dimensions=2)
        coords = ordi.samples.iloc[:, :2].values; expl = ordi.proportion_explained.iloc[:2].values * 100
        for grp in ["autism", "control"]:
            m = gsub.values == grp
            ax.scatter(coords[m, 0], coords[m, 1], s=24, alpha=0.75, c=PALETTE[grp], label=grp)
        ax.set_xlabel(f"PC1 ({expl[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({expl[1]:.0f}%)")
        ax.set_title(f"{agl} (n={len(idx)}; PERMANOVA p={pe['p-value']:.3f})", fontsize=9)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle("Functional pathways: PCoA + PERMANOVA within each age group")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pw_beta_by_agegroup.png"), dpi=140); plt.close(fig)
    pd.DataFrame(brows).to_csv(os.path.join(OUT, "pw_permanova_by_agegroup.csv"), index=False)
    log.info("functional PERMANOVA within age groups: %s",
             {r["age_group"]: round(r["PERMANOVA_p"], 3) for r in brows})

    # per-stratum differential pathways (CLR-LM) + stratified effect heatmap
    strat = {agl: C.differential_lm(clr.loc[cov_ag.index[cov_ag.age_group == agl]],
                                    cov.loc[cov_ag.index[cov_ag.age_group == agl]]) for agl in AGES}
    nsig_strat = {agl: int((r["qval"] < 0.05).sum()) if not r.empty else 0 for agl, r in strat.items()}
    log.info("functional differential pathways significant (q<0.05) per age group: %s", nsig_strat)
    focus = list(da.reindex(da["coef_group"].abs().sort_values(ascending=False).index).head(20).index)
    for r in strat.values():
        if not r.empty:
            focus += list(r[r["qval"] < 0.05].index)
    focus = list(dict.fromkeys(focus))[:24]
    if focus:
        coef_tab = pd.DataFrame({"overall": da["coef_group"].reindex(focus)}, index=focus)
        sig_tab = pd.DataFrame({"overall": (da["qval"].reindex(focus) < 0.05)}, index=focus)
        for agl, r in strat.items():
            coef_tab[agl] = r["coef_group"].reindex(focus) if not r.empty else np.nan
            sig_tab[agl] = (r["qval"].reindex(focus) < 0.05) if not r.empty else False
        coef_tab.to_csv(os.path.join(OUT, "pw_differential_stratified.csv"))
        fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(focus))))
        im = ax.imshow(coef_tab.values.astype(float), cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(coef_tab.shape[1])); ax.set_xticklabels(coef_tab.columns)
        ax.set_yticks(range(len(focus))); ax.set_yticklabels([t.split(":")[0][:30] for t in focus], fontsize=6)
        for (i, j), v in np.ndenumerate(coef_tab.values.astype(float)):
            if bool(sig_tab.values[i, j]):
                ax.text(j, i, "*", ha="center", va="center", color="k", fontsize=8)
        ax.set_title("Functional pathways: CLR coef (autism−control) by age group\n"
                     "(* q<0.05; columns: overall + 3 strata)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.5, label="CLR coef")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "pw_differential_stratified.png"), dpi=150); plt.close(fig)

    log.info("functional-pathway analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
