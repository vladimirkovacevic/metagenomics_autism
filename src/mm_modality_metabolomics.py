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

    log.info("functional-pathway analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
