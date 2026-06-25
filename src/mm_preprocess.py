"""Preprocessing for multi-omics analysis (PLAN_MM.md §3).

Reads the three harmonized matrices and writes analysis-ready, transformed
matrices to results/preprocessed/:

  covariates.tsv              age (2024-birthyear), sex, group  [+ K19/K23 flag]
  metagenomics_filt.tsv       prevalence/abundance-filtered relative abundance
  metagenomics_clr.tsv        CLR of the filtered metagenomics
  metabolomics_filt.tsv       TSS relative abundance, UNMAPPED/UNINTEGRATED dropped, filtered
  metabolomics_clr.tsv        CLR of the filtered metabolomics
  clinical_analysis.tsv       broad clinical set for univariate tests (continuous+categorical, raw)
  clinical_scores.tsv         clean continuous summary scores (SSDS/SDQ/GIRBI) for integration
  clinical_types.tsv          variable -> continuous/categorical
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import mm_common as C

log = C.get_logger("preprocess")

PREVALENCE = 0.10        # keep features present in >=10% of samples
MIN_MEAN_ABUND = 1e-4    # metagenomics mean relative-abundance floor
AGE_FLAG_SAMPLES = ["K19", "K23"]  # birth-year vs Uzrast inconsistency (PLAN_MM.md §0)


def filter_compositional(df: pd.DataFrame, prevalence: float, min_mean: float | None):
    present = (df > 0).mean(axis=0)
    keep = present >= prevalence
    if min_mean is not None:
        keep &= df.mean(axis=0) >= min_mean
    return df.loc[:, keep]


def main():
    C.ensure_dirs(C.PRE)
    mg, mb, cl = C.load_harmonized()
    log.info("loaded harmonized: metagenomics %s, metabolomics %s, clinical %s",
             mg.shape, mb.shape, cl.shape)

    # ----- covariates ----------------------------------------------------- #
    age = C.derive_age(cl)
    sex = C.get_sex(cl)
    group = C.get_group(cl)
    cov = pd.concat([group, age, sex], axis=1)
    cov["age_flag"] = [s in AGE_FLAG_SAMPLES for s in cov.index]
    cov.to_csv(os.path.join(C.PRE, "covariates.tsv"), sep="\t")
    log.info("covariates: %d samples | age missing=%d | sex missing=%d | flagged=%s",
             len(cov), int(age.isna().sum()), int(sex.isna().sum()), AGE_FLAG_SAMPLES)
    log.info("group counts: %s", group.value_counts().to_dict())

    # ----- metagenomics counts (for DESeq2; integer reads from bracken_num) - #
    raw = pd.read_csv(os.path.join(C.DATA, "merged_bracken.tsv"), sep="\t", low_memory=False)
    num_cols = {c[:-len(".bracken_num")]: c for c in raw.columns if c.endswith(".bracken_num")}
    counts = raw.set_index("name")[[num_cols[s] for s in mg.index]].copy()
    counts.columns = list(mg.index)               # samples in harmonized order
    counts = counts.round().astype(int)
    counts.index.name = "taxon"
    counts.to_csv(os.path.join(C.PRE, "metagenomics_counts.tsv"), sep="\t")
    log.info("metagenomics counts (for DESeq2): %d taxa x %d samples", *counts.shape)

    # ----- metagenomics (relative abundance / CLR) ------------------------ #
    mg_filt = filter_compositional(mg, PREVALENCE, MIN_MEAN_ABUND)
    mg_filt = mg_filt.div(mg_filt.sum(axis=1), axis=0)  # re-close
    log.info("metagenomics: %d -> %d taxa after prevalence>=%.0f%% & mean>=%.0e",
             mg.shape[1], mg_filt.shape[1], PREVALENCE * 100, MIN_MEAN_ABUND)
    mg_clr = C.clr_transform(mg_filt)
    mg_filt.to_csv(os.path.join(C.PRE, "metagenomics_filt.tsv"), sep="\t")
    mg_clr.to_csv(os.path.join(C.PRE, "metagenomics_clr.tsv"), sep="\t")

    # ----- metabolomics (functional pathways) ----------------------------- #
    drop = [c for c in mb.columns if c in ("UNMAPPED", "UNINTEGRATED")]
    mb2 = mb.drop(columns=drop)
    log.info("metabolomics: dropped %s; %d pathways remain", drop, mb2.shape[1])
    mb_tss = mb2.div(mb2.sum(axis=1), axis=0)            # total-sum scaling
    mb_filt = filter_compositional(mb_tss, PREVALENCE, None)
    mb_filt = mb_filt.div(mb_filt.sum(axis=1), axis=0)
    log.info("metabolomics: %d -> %d pathways after prevalence>=%.0f%%",
             mb2.shape[1], mb_filt.shape[1], PREVALENCE * 100)
    mb_clr = C.clr_transform(mb_filt)
    mb_filt.to_csv(os.path.join(C.PRE, "metabolomics_filt.tsv"), sep="\t")
    mb_clr.to_csv(os.path.join(C.PRE, "metabolomics_clr.tsv"), sep="\t")

    # ----- clinical ------------------------------------------------------- #
    continuous, categorical, dropped = C.classify_clinical_columns(cl)
    log.info("clinical: %d continuous, %d categorical, %d dropped",
             len(continuous), len(categorical), len(dropped))
    for c, reason in list(dropped.items())[:5]:
        log.info("clinical drop e.g. %s -> %s", c, reason)

    # broad analysis set (raw values; univariate tests handle missingness/types)
    analysis = cl[continuous + categorical].copy()
    analysis.insert(0, "group", group)
    analysis.insert(1, "age", age)
    analysis.insert(2, "sex", sex.map({"M": 1, "F": 0}))
    analysis.to_csv(os.path.join(C.PRE, "clinical_analysis.tsv"), sep="\t")

    types = pd.Series(
        {**{c: "continuous" for c in continuous}, **{c: "categorical" for c in categorical}},
        name="type")
    types.to_csv(os.path.join(C.PRE, "clinical_types.tsv"), sep="\t")

    # clean continuous summary scores for integration (SSDS/SDQ/GIRBI + age)
    score_cols = [c for c in C.CLINICAL_SCORE_BLOCK if c in cl.columns]
    scores = cl[score_cols].apply(pd.to_numeric, errors="coerce")
    scores.insert(0, "age", age)
    log.info("clinical score block for integration: %d scores (+age), %d samples",
             len(score_cols), len(scores))
    scores.to_csv(os.path.join(C.PRE, "clinical_scores.tsv"), sep="\t")

    log.info("preprocessing complete -> %s", os.path.relpath(C.PRE, C.ROOT))


if __name__ == "__main__":
    main()
