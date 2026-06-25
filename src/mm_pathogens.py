"""Annotate DESeq2 differential taxa against a curated pathogen reference
(PLAN_MM.md §4.1).

No internet access is available, so we match the differentially abundant species
against a curated, offline reference list of recognised human enteric /
opportunistic pathogens (genus- and species-level; compiled from standard
clinical-microbiology and PATRIC/Virulence-factor sources). Every differential
taxon is reported with its pathogen status. Output -> results/metagenomics/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mm_common as C

C.set_style()
log = C.get_logger("pathogens")
OUT = os.path.join(C.RESULTS, "metagenomics")

# Curated offline reference of recognised human gut/enteric and opportunistic
# pathogens. Genus-level entries flag the genus; "Genus species" entries are
# matched as a stricter species hit.
PATHOGEN_GENERA = {
    "Salmonella", "Shigella", "Campylobacter", "Helicobacter", "Vibrio",
    "Yersinia", "Listeria", "Clostridioides", "Clostridium", "Klebsiella",
    "Pseudomonas", "Proteus", "Enterobacter", "Citrobacter", "Morganella",
    "Serratia", "Acinetobacter", "Staphylococcus", "Enterococcus", "Fusobacterium",
    "Aeromonas", "Plesiomonas", "Bilophila", "Sutterella", "Campylobacterales",
}
PATHOGEN_SPECIES = {
    "escherichia coli", "bacteroides fragilis", "streptococcus pyogenes",
    "streptococcus pneumoniae", "streptococcus agalactiae", "klebsiella pneumoniae",
    "enterococcus faecalis", "enterococcus faecium", "staphylococcus aureus",
    "clostridioides difficile", "clostridium perfringens", "pseudomonas aeruginosa",
    "fusobacterium nucleatum", "helicobacter pylori", "listeria monocytogenes",
}


def pathogen_status(taxon: str) -> str:
    t = str(taxon).strip().lower()
    if any(sp in t for sp in PATHOGEN_SPECIES):
        return "pathogen (species)"
    genus = str(taxon).split()[0] if taxon else ""
    if genus in PATHOGEN_GENERA:
        return "pathogen (genus)"
    return "commensal/other"


def main():
    path = os.path.join(OUT, "deseq2_results.csv")
    if not os.path.exists(path):
        log.error("deseq2_results.csv not found — run mm_deseq2.R first"); return
    res = pd.read_csv(path)
    res["pathogen_status"] = res["taxon"].map(pathogen_status)
    res["is_pathogen"] = res["pathogen_status"] != "commensal/other"
    res.to_csv(os.path.join(OUT, "deseq2_pathogen_annotated.csv"), index=False)

    sig_all = res[res["padj"] < 0.05].copy()
    n_path = int(sig_all["is_pathogen"].sum())
    log.info("DESeq2: %d significant taxa (padj<0.05); %d flagged as known pathogens",
             len(sig_all), n_path)
    for _, r in sig_all[sig_all["is_pathogen"]].iterrows():
        log.info("  pathogen hit: %s (log2FC=%+.2f, padj=%.3g, %s)",
                 r["taxon"], r["log2FoldChange"], r["padj"], r["pathogen_status"])

    # figure: top significant taxa by |log2FC| (all sig pathogens forced in), pathogens outlined
    top = sig_all.reindex(sig_all["log2FoldChange"].abs().sort_values(ascending=False).index).head(28)
    sig = pd.concat([top, sig_all[sig_all["is_pathogen"]]]).drop_duplicates("taxon")
    sig = sig.sort_values("log2FoldChange")
    if len(sig):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(sig))))
        colors = np.where(sig["log2FoldChange"] > 0, C.SIG_UP, C.SIG_DOWN)
        edge = np.where(sig["is_pathogen"], "black", "none")
        lw = np.where(sig["is_pathogen"], 1.8, 0.0)
        ax.barh(range(len(sig)), sig["log2FoldChange"], color=colors,
                edgecolor=edge, linewidth=list(lw))
        labels = [f"{'★ ' if p else ''}{t[:40]}"
                  for t, p in zip(sig["taxon"], sig["is_pathogen"])]
        ax.set_yticks(range(len(sig))); ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel("DESeq2 log2 fold change (autism vs control)")
        ax.set_title(f"DESeq2 differential taxa: top {len(sig)} of {len(sig_all)} sig "
                     f"(★ = known pathogen, {n_path} total)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "deseq2_pathogens.png"), dpi=160)
        plt.close(fig)
    log.info("pathogen annotation complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
