"""Validate DESeq2 pathogen hits against the PATRIC / BV-BRC database (PLAN_MM.md §4.1).

For every DESeq2-significant taxon we query the live BV-BRC (PATRIC) genome API
and use the curated `disease` metadata field as the confirmation signal: a
species with >=1 genome carrying a non-empty `disease` annotation is a
DB-confirmed disease-associated organism (commensals return zero; verified on
K. pneumoniae / C. perfringens / S. aureus vs Blautia / Faecalibacterium).

This upgrades the earlier offline genus-level flag (loose) to a species-level,
database-backed validation, and visualises the outcome. Results are cached to
results/metagenomics/patric_cache.tsv so re-runs are offline/instant.
Outputs -> results/metagenomics/{pathogen_validation.csv, pathogen_validation.png}.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mm_common as C

C.set_style()
log = C.get_logger("patric")
OUT = os.path.join(C.RESULTS, "metagenomics")
CACHE = os.path.join(OUT, "patric_cache.tsv")
API = "https://www.bv-brc.org/api/genome/"
MAX_QUERIES = 260


def clean_species(taxon: str):
    """Reduce a Bracken taxon name to a queryable 'Genus species' binomial, or None."""
    t = re.sub(r"[\[\]]", "", str(taxon)).strip()
    toks = t.split()
    if toks and toks[0] == "Candidatus":
        toks = toks[1:]
    if len(toks) < 2:
        return None
    genus, epithet = toks[0], toks[1]
    if epithet.lower() in {"sp.", "sp", "bacterium", "endosymbiont", "cf."} or not epithet[:1].islower():
        return None  # strain / unresolved (e.g. "Enterobacter sp. T2")
    return f"{genus} {epithet}"


def query_patric(species: str):
    """Return (n_genomes, disease_records, top_disease) for a species from BV-BRC."""
    rql = (f"eq(species,{urllib.parse.quote(species)})"
           f"&select(disease,host_name)&limit(200)")
    req = urllib.request.Request(API + "?" + rql, headers={"Accept": "application/json"})
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        log.warning("query failed for %s: %s", species, e)
        return None
    diseases = []
    for x in data:
        v = x.get("disease")
        if isinstance(v, list):
            diseases += [str(i).strip() for i in v if i]
        elif v:
            diseases.append(str(v).strip())
    top = pd.Series(diseases).value_counts().index[0] if diseases else ""
    return len(data), len(diseases), top


def load_cache():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, sep="\t", index_col=0).to_dict("index")
    return {}


def main():
    res = pd.read_csv(os.path.join(OUT, "deseq2_pathogen_annotated.csv"))
    sig = res[res["padj"] < 0.05].copy()
    sig["query_species"] = sig["taxon"].map(clean_species)
    log.info("validating %d DESeq2-significant taxa against PATRIC (%d resolvable to species)",
             len(sig), sig["query_species"].notna().sum())

    cache = load_cache()
    to_query = [s for s in sig["query_species"].dropna().unique() if s not in cache]
    if len(to_query) > MAX_QUERIES:
        log.warning("capping PATRIC queries at %d (of %d); rest left unresolved", MAX_QUERIES, len(to_query))
        to_query = to_query[:MAX_QUERIES]
    for i, sp in enumerate(to_query):
        r = query_patric(sp)
        if r is not None:
            cache[sp] = {"n_genomes": r[0], "disease_records": r[1], "top_disease": r[2]}
            time.sleep(0.05)
        if (i + 1) % 25 == 0:
            log.info("  queried %d/%d ...", i + 1, len(to_query))
    pd.DataFrame.from_dict(cache, orient="index").to_csv(CACHE, sep="\t")
    log.info("PATRIC cache holds %d species", len(cache))

    def status(sp):
        if sp is None:
            return "unresolved (strain/sp.)", 0, ""
        c = cache.get(sp)
        if c is None or c["n_genomes"] == 0:
            return "not in PATRIC", 0, ""
        if c["disease_records"] > 0:
            return "PATRIC disease-confirmed", int(c["disease_records"]), c.get("top_disease", "")
        return "in PATRIC, no disease record", 0, ""

    st = sig["query_species"].map(lambda s: status(s))
    sig["patric_status"] = [x[0] for x in st]
    sig["patric_disease_records"] = [x[1] for x in st]
    sig["patric_top_disease"] = [x[2] for x in st]
    cols = ["taxon", "log2FoldChange", "padj", "is_pathogen", "pathogen_status",
            "query_species", "patric_status", "patric_disease_records", "patric_top_disease"]
    sig[cols].sort_values("patric_disease_records", ascending=False).to_csv(
        os.path.join(OUT, "pathogen_validation.csv"), index=False)

    confirmed = sig[sig["patric_status"] == "PATRIC disease-confirmed"]
    n_conf = len(confirmed)
    n_flag_conf = int(((sig["is_pathogen"]) & (sig["patric_status"] == "PATRIC disease-confirmed")).sum())
    n_flag = int(sig["is_pathogen"].sum())
    log.info("PATRIC-confirmed disease-associated species among DESeq2 hits: %d", n_conf)
    log.info("offline-flagged pathogens confirmed by PATRIC: %d / %d", n_flag_conf, n_flag)

    # ---------------- visualisation ---------------- #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(5, 0.34 * max(len(confirmed), 8))),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    # Panel A: confirmed pathogens, effect size + top disease
    cf = confirmed.sort_values("log2FoldChange")
    if len(cf):
        colors = np.where(cf["log2FoldChange"] > 0, C.SIG_UP, C.SIG_DOWN)
        ax1.hlines(range(len(cf)), 0, cf["log2FoldChange"], color=colors, lw=2, zorder=1)
        ax1.scatter(cf["log2FoldChange"], range(len(cf)), s=30 + 6 * cf["patric_disease_records"],
                    color=colors, zorder=2)
        labels = [f"{t.split()[0][:14]} {t.split()[1] if len(t.split())>1 else ''}"
                  f"  [{d}]" for t, d in zip(cf["query_species"].fillna(cf["taxon"]),
                                             cf["patric_top_disease"])]
        ax1.set_yticks(range(len(cf))); ax1.set_yticklabels(labels, fontsize=7)
        ax1.axvline(0, c="k", lw=0.6)
    ax1.set_xlabel("DESeq2 log2 fold change (autism − control)")
    ax1.set_title(f"PATRIC disease-confirmed pathogens among DESeq2 hits (n={n_conf})\n"
                  "[top disease]; red ↑autism, blue ↑control; size ∝ disease records", fontsize=9)

    # Panel B: validation outcome of the offline flags
    order = ["PATRIC disease-confirmed", "in PATRIC, no disease record",
             "not in PATRIC", "unresolved (strain/sp.)"]
    flagged = sig[sig["is_pathogen"]]
    counts = flagged["patric_status"].value_counts().reindex(order).fillna(0)
    bar_c = ["#1a9850", "#a6d96a", "#fdae61", "#B0B0B0"]
    ax2.barh(range(len(order))[::-1], counts.values, color=bar_c)
    ax2.set_yticks(range(len(order))[::-1]); ax2.set_yticklabels(order, fontsize=8)
    for i, v in zip(range(len(order))[::-1], counts.values):
        ax2.text(v + 0.3, i, str(int(v)), va="center", fontsize=8)
    ax2.set_xlabel("offline-flagged significant taxa")
    ax2.set_title(f"Validation of offline genus-level flags\n({n_flag_conf}/{n_flag} confirmed by PATRIC)",
                  fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pathogen_validation.png"), dpi=150); plt.close(fig)
    log.info("pathogen validation complete -> %s", os.path.relpath(OUT, C.ROOT))


if __name__ == "__main__":
    main()
