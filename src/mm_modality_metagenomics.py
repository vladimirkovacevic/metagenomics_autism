"""Metagenomics modality analysis (PLAN_MM.md §4.1).

Alpha diversity, beta diversity / ordination + PERMANOVA, and age/sex-adjusted
differential abundance. Figures (PNG) + result tables (CSV) -> results/metagenomics/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

import mm_common as C

log = C.get_logger("metagenomics")
OUT = os.path.join(C.RESULTS, "metagenomics")
C.set_style()
PALETTE = C.GROUP_COLORS


def alpha_diversity(rel: pd.DataFrame) -> pd.DataFrame:
    p = rel.div(rel.sum(axis=1), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        shannon = -(p * np.log(p)).replace([-np.inf, np.inf], 0).fillna(0).sum(axis=1)
    simpson = 1 - (p ** 2).sum(axis=1)
    richness = (rel > 0).sum(axis=1)
    return pd.DataFrame({"Shannon": shannon, "Simpson": simpson, "Richness": richness})


def adjusted_group_test(metric: pd.Series, cov: pd.DataFrame):
    """LM: metric ~ group + age + sex; return group coefficient and p-value."""
    import statsmodels.api as sm
    d = C.build_design(cov)
    df = d.join(metric.rename("y")).dropna()
    X = sm.add_constant(df[["group01", "age", "sex01"]])
    res = sm.OLS(df["y"], X).fit()
    return res.params["group01"], res.pvalues["group01"]


def main():
    C.ensure_dirs(OUT)
    mg_filt = pd.read_csv(os.path.join(C.PRE, "metagenomics_filt.tsv"), sep="\t", index_col=0)
    mg_clr = pd.read_csv(os.path.join(C.PRE, "metagenomics_clr.tsv"), sep="\t", index_col=0)
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)
    group = cov["group"].astype(str)
    log.info("metagenomics analysis: %d samples x %d taxa", *mg_filt.shape)

    # ---- alpha diversity ---- #
    alpha = alpha_diversity(mg_filt)
    alpha.join(group).to_csv(os.path.join(OUT, "alpha_diversity.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    alpha_stats = {}
    for ax, metric in zip(axes, alpha.columns):
        data = [alpha.loc[group == g, metric] for g in ["autism", "control"]]
        bp = ax.boxplot(data, labels=["autism", "control"], patch_artist=True, widths=0.6)
        for patch, g in zip(bp["boxes"], ["autism", "control"]):
            patch.set_facecolor(PALETTE[g]); patch.set_alpha(0.7)
        coef, p = adjusted_group_test(alpha[metric], cov)
        alpha_stats[metric] = {"group_coef_adj": coef, "pval_adj": p}
        ax.set_title(f"{metric}\n(age/sex-adj p={p:.3f})")
    fig.suptitle("Alpha diversity by group (autism vs control)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "alpha_diversity.png"), dpi=130); plt.close(fig)
    pd.DataFrame(alpha_stats).T.to_csv(os.path.join(OUT, "alpha_diversity_stats.csv"))
    log.info("alpha diversity adjusted p-values: %s",
             {k: round(v["pval_adj"], 4) for k, v in alpha_stats.items()})

    # ---- beta diversity: Bray-Curtis (rel) & Aitchison (euclidean on CLR) ---- #
    perm_results = {}
    for name, dist in [("BrayCurtis", pdist(mg_filt.values, metric="braycurtis")),
                       ("Aitchison", pdist(mg_clr.values, metric="euclidean"))]:
        dm = DistanceMatrix(squareform(dist), ids=list(mg_filt.index))
        ordi = pcoa(dm, number_of_dimensions=2)
        coords = ordi.samples.iloc[:, :2]
        coords.columns = ["PC1", "PC2"]
        pe = permanova(dm, grouping=list(group), permutations=999)
        perm_results[name] = {"pseudo_F": pe["test statistic"], "pval": pe["p-value"]}
        expl = ordi.proportion_explained.iloc[:2].values * 100
        fig, ax = plt.subplots(figsize=(6, 5))
        for g in ["autism", "control"]:
            m = group.values == g
            ax.scatter(coords.values[m, 0], coords.values[m, 1], s=28, alpha=0.7,
                       c=PALETTE[g], label=g)
        ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
        ax.set_title(f"{name} PCoA — PERMANOVA F={pe['test statistic']:.2f}, p={pe['p-value']:.3f}")
        ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"pcoa_{name}.png"), dpi=130); plt.close(fig)
    pd.DataFrame(perm_results).T.to_csv(os.path.join(OUT, "permanova.csv"))
    log.info("PERMANOVA (group): %s", {k: round(v["pval"], 4) for k, v in perm_results.items()})

    # ---- differential abundance (CLR-LM, age/sex-adjusted, BH-FDR) ---- #
    log.info("differential abundance via CLR linear models (ANCOM-BC substitute, see PLAN_MM.md D1)")
    da = C.differential_lm(mg_clr, cov)
    da.to_csv(os.path.join(OUT, "differential_abundance.csv"))
    sig = da[da["qval"] < 0.05]
    log.info("differential taxa: %d significant at q<0.05 (of %d tested)", len(sig), len(da))

    # volcano
    fig, ax = plt.subplots(figsize=(7, 5))
    nlq = -np.log10(da["qval"].clip(lower=1e-300))
    colors = [C.sig_color(c, q) for c, q in zip(da["coef_group"], da["qval"])]
    ax.scatter(da["coef_group"], nlq, s=16, c=colors, alpha=0.8, edgecolors="none")
    ax.axhline(-np.log10(0.05), ls="--", c="k", lw=0.8)
    ax.set_xlabel("CLR coefficient (autism − control, age/sex-adj)")
    ax.set_ylabel("-log10(q)")
    ax.set_title(f"Differential taxa volcano ({len(sig)} sig at q<0.05)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "differential_volcano.png"), dpi=130); plt.close(fig)

    # top effect-size barplot
    top = da.reindex(da["coef_group"].abs().sort_values(ascending=False).index).head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(top)), top["coef_group"],
            color=np.where(top["coef_group"] > 0, C.SIG_UP, C.SIG_DOWN))
    ax.set_yticks(range(len(top))); ax.set_yticklabels([t[:45] for t in top.index], fontsize=7)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("CLR coefficient (autism − control)")
    ax.set_title("Top 20 taxa by effect size (red=↑autism, blue=↑control)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "differential_top_taxa.png"), dpi=130); plt.close(fig)

    log.info("metagenomics analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
