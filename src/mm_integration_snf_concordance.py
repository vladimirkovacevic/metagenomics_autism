"""SNF, cross-omics concordance, and within-autism sPLS (PLAN_MM.md §5.3).

  * SNF (Similarity Network Fusion) of metagenomics + pathways -> spectral
    clusters -> association with group/age; fused-network heatmap.
  * Concordance: Mantel test + Procrustes between the two omics distance spaces;
    PLS canonical correlation (cross-group).
  * Within-autism sPLS: relate metagenomics+pathways (X) to ASD symptom scores
    SSDS/SDQ/GIRBI (Y) in the autism subset.
Outputs -> results/integration/snf/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.cluster import SpectralClustering
from sklearn.cross_decomposition import PLSCanonical, PLSRegression
from skbio import DistanceMatrix
from skbio.stats.distance import mantel
from skbio.stats.ordination import pcoa

import mm_common as C

log = C.get_logger("snf")
OUT = os.path.join(C.RESULTS, "integration", "snf")
C.set_style()
PALETTE = C.GROUP_COLORS


def affinity(X, K=20, mu=0.5):
    """Scaled-exponential local affinity matrix (Wang et al. SNF)."""
    D = squareform(pdist(X, metric="euclidean"))
    n = D.shape[0]
    knn_mean = np.sort(D, axis=1)[:, 1:K + 1].mean(axis=1)
    eps = (knn_mean[:, None] + knn_mean[None, :]) / 2 + 1e-10
    W = np.exp(-(D ** 2) / (mu * eps ** 2))
    np.fill_diagonal(W, 0)
    return W


def _norm(W):
    s = W.sum(axis=1, keepdims=True); s[s == 0] = 1
    return W / s


def _sparse_knn(P, K=20):
    n = P.shape[0]; S = np.zeros_like(P)
    idx = np.argsort(-P, axis=1)[:, :K]
    for i in range(n):
        S[i, idx[i]] = P[i, idx[i]]
    return _norm(S)


def snf(views, K=20, t=20):
    Ps = [_norm(affinity(v, K)) for v in views]
    Ss = [_sparse_knn(p, K) for p in Ps]
    for _ in range(t):
        newP = []
        for i in range(len(Ps)):
            others = sum(Ps[j] for j in range(len(Ps)) if j != i) / (len(Ps) - 1)
            newP.append(Ss[i] @ others @ Ss[i].T)
        Ps = [_norm(p) for p in newP]
    return sum(Ps) / len(Ps)


def main():
    C.ensure_dirs(OUT)
    mg = pd.read_csv(os.path.join(C.PRE, "metagenomics_clr.tsv"), sep="\t", index_col=0)
    mb = pd.read_csv(os.path.join(C.PRE, "metabolomics_clr.tsv"), sep="\t", index_col=0)
    cov = pd.read_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t", index_col=0)
    group = cov["group"].astype(str)
    samples = list(mg.index)

    # =================== SNF ==================== #
    W = snf([mg.values, mb.values], K=20, t=20)
    pd.DataFrame(W, index=samples, columns=samples).to_csv(os.path.join(OUT, "fused_network.csv"))
    n_clusters = 2
    labels = SpectralClustering(n_clusters=n_clusters, affinity="precomputed",
                                random_state=1).fit_predict(W)
    clusters = pd.Series(labels, index=samples, name="cluster")
    ct = pd.crosstab(clusters, group)
    chi2_p = stats.chi2_contingency(ct.values)[1]
    age = pd.to_numeric(cov["age"], errors="coerce")
    age_p = stats.kruskal(*[age[clusters == c].dropna() for c in range(n_clusters)])[1]
    clusters.to_frame().join(group).to_csv(os.path.join(OUT, "snf_clusters.csv"))
    log.info("SNF clusters vs group: %s (chi2 p=%.3g) | vs age kruskal p=%.3g",
             ct.to_dict(), chi2_p, age_p)

    order = np.argsort(labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(W[np.ix_(order, order)], cmap="magma")
    ax.set_title(f"SNF fused similarity network (spectral clusters)\n"
                 f"cluster~group chi2 p={chi2_p:.3f}, cluster~age p={age_p:.3f}")
    ax.set_xticks([]); ax.set_yticks([])
    rc = [PALETTE[group.iloc[i]] for i in order]
    ax.scatter(range(len(order)), [-2] * len(order), c=rc, s=6, marker="s", clip_on=False)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fused similarity")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "snf_network.png"), dpi=130); plt.close(fig)

    # =============== concordance: Mantel + Procrustes =============== #
    d_mg = DistanceMatrix(squareform(pdist(mg.values)), ids=samples)
    d_mb = DistanceMatrix(squareform(pdist(mb.values)), ids=samples)
    r_mantel, p_mantel, _ = mantel(d_mg, d_mb, permutations=999, method="spearman")
    coords_mg = pcoa(d_mg, number_of_dimensions=5).samples.values
    coords_mb = pcoa(d_mb, number_of_dimensions=5).samples.values
    _, _, m2 = procrustes(coords_mg, coords_mb)  # m2 = disparity (0=identical)
    log.info("concordance: Mantel r=%.3f p=%.3g | Procrustes disparity=%.3f (lower=more concordant)",
             r_mantel, p_mantel, m2)

    # PLS canonical correlation (cross-group), top variable features for speed
    mgv = mg[mg.var().sort_values(ascending=False).head(100).index]
    mbv = mb[mb.var().sort_values(ascending=False).head(100).index]
    pls = PLSCanonical(n_components=2).fit(mgv.values, mbv.values)
    xs, ys = pls.transform(mgv.values, mbv.values)
    canon_r = [stats.pearsonr(xs[:, i], ys[:, i])[0] for i in range(2)]
    log.info("PLS canonical correlations (metagenomics<->pathways): %s",
             [round(r, 3) for r in canon_r])
    pd.DataFrame({"metric": ["mantel_r", "mantel_p", "procrustes_disparity",
                             "PLS_canon_r1", "PLS_canon_r2"],
                  "value": [r_mantel, p_mantel, m2, canon_r[0], canon_r[1]]}
                 ).to_csv(os.path.join(OUT, "concordance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    for g in ["autism", "control"]:
        m = group.values == g
        ax.scatter(xs[m, 0], ys[m, 0], s=26, alpha=0.7, c=PALETTE[g], label=g)
    ax.set_xlabel("metagenomics PLS dim1"); ax.set_ylabel("pathways PLS dim1")
    ax.set_title(f"Cross-omics PLS (canonical r={canon_r[0]:.2f}); Mantel r={r_mantel:.2f}, p={p_mantel:.3f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "concordance_pls.png"), dpi=130); plt.close(fig)

    # =============== within-autism sPLS to symptom scores =============== #
    scores = pd.read_csv(os.path.join(C.PRE, "clinical_scores.tsv"), sep="\t", index_col=0)
    score_cols = [c for c in C.CLINICAL_SCORE_BLOCK if c in scores.columns]
    a_idx = group.index[group == "autism"]
    Y = scores.loc[a_idx, score_cols].apply(pd.to_numeric, errors="coerce")
    # keep autism samples with >=50% scores present; median-impute remaining
    Y = Y.loc[Y.notna().mean(axis=1) >= 0.5]
    Y = Y.fillna(Y.median())
    X = pd.concat([mg.loc[Y.index].add_prefix("MG:"), mb.loc[Y.index].add_prefix("PW:")], axis=1)
    log.info("within-autism sPLS: %d autism samples, X=%d features, Y=%d scores",
             X.shape[0], X.shape[1], Y.shape[1])
    pls2 = PLSRegression(n_components=2).fit(X.values, Y.values)
    # correlate each X feature's component-1 weight-projection with each score
    xscore = pls2.x_scores_[:, 0]
    feat_corr = pd.Series(
        {f: stats.pearsonr(X[f].values, xscore)[0] for f in X.columns}).sort_values()
    top_feats = pd.concat([feat_corr.head(15), feat_corr.tail(15)]).index.tolist()
    corr_mat = pd.DataFrame(
        {sc: [stats.pearsonr(X[f].values, Y[sc].values)[0] for f in top_feats] for sc in score_cols},
        index=top_feats)
    corr_mat.to_csv(os.path.join(OUT, "within_autism_feature_score_corr.csv"))

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(score_cols)))
    ax.set_xticklabels([c[:28] for c in score_cols], rotation=90, fontsize=7)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels([f.split(":")[0] + ":" + f.split(":")[-1].split(" ")[0][:30] for f in top_feats],
                       fontsize=6)
    ax.set_title("Within-autism: top omics features vs ASD symptom scores (Pearson r)")
    fig.colorbar(im, ax=ax, shrink=0.5, label="r")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "within_autism_spls.png"), dpi=130); plt.close(fig)

    # =============== differential taxon co-abundance network =============== #
    # Pearson correlation networks of the most variable taxa, computed separately
    # in autism and control, then compared edge-by-edge.
    K, THR = 40, 0.4
    top_taxa = mg.var().sort_values(ascending=False).head(K).index
    Xa = mg.loc[group == "autism", top_taxa]
    Xc = mg.loc[group == "control", top_taxa]
    Ca, Cc = Xa.corr().values, Xc.corr().values
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    pos = np.c_[np.cos(ang), np.sin(ang)]

    def draw_net(ax, Cm, title):
        ne = 0
        for i in range(K):
            for j in range(i + 1, K):
                if abs(Cm[i, j]) >= THR:
                    ax.plot(pos[[i, j], 0], pos[[i, j], 1], lw=0.4 + 2 * (abs(Cm[i, j]) - THR),
                            color=(C.SIG_UP if Cm[i, j] > 0 else C.SIG_DOWN), alpha=0.5, zorder=1)
                    ne += 1
        ax.scatter(pos[:, 0], pos[:, 1], s=18, color=C.CAT_PALETTE[0], zorder=2)
        ax.set_title(f"{title} ({ne} edges)"); ax.axis("off"); ax.set_aspect("equal")
        return ne

    na = (np.abs(np.triu(Ca, 1)) >= THR)
    nc = (np.abs(np.triu(Cc, 1)) >= THR)
    shared = int((na & nc).sum()); a_only = int((na & ~nc).sum()); c_only = int((nc & ~na).sum())
    # biggest rewiring: largest |r_autism - r_control| among taxon pairs
    diff = np.triu(Ca - Cc, 1)
    iu = np.triu_indices(K, 1)
    dpairs = sorted(zip(np.abs(diff[iu]), top_taxa[iu[0]], top_taxa[iu[1]],
                        Ca[iu], Cc[iu]), reverse=True)[:15]
    pd.DataFrame(dpairs, columns=["abs_delta_r", "taxon_a", "taxon_b", "r_autism", "r_control"]
                 ).to_csv(os.path.join(OUT, "differential_network_top.csv"), index=False)
    pd.Series({"edges_autism": int(na.sum()), "edges_control": int(nc.sum()),
               "shared": shared, "autism_only": a_only, "control_only": c_only}
              ).to_csv(os.path.join(OUT, "differential_network_stats.csv"))
    log.info("differential network (|r|>=%.1f): autism=%d, control=%d edges; shared=%d, "
             "autism-only=%d, control-only=%d", THR, int(na.sum()), int(nc.sum()),
             shared, a_only, c_only)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    draw_net(axes[0], Ca, "Autism")
    draw_net(axes[1], Cc, "Control")
    fig.suptitle(f"Taxon co-abundance networks ({K} most-variable taxa, |Pearson r|>={THR}; "
                 f"red +, blue −)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "differential_network.png"), dpi=150); plt.close(fig)

    log.info("SNF/concordance/within-autism analysis complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
