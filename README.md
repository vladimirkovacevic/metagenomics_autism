# metagenomics_autism

A reproducible **multi-omic analysis pipeline** comparing the gut microbiome of
children with autism spectrum disorder (ASD) and matched controls. It harmonizes
shotgun-metagenomic taxonomy, microbial functional pathways, and clinical
records into sample-aligned matrices, runs per-modality and integrative
analyses, and assembles a multi-panel PDF report.

> **Data is not included in this repository.** Place your own input files under
> `data/` (see *Inputs* below). All `data/`, `results/`, and environment folders
> are git-ignored.

---

## What it does

1. **Harmonization** (`src/harmonization.py`) — links sample IDs across
   platforms, derives age, de-duplicates, and aligns three datasets to the same
   samples in identical order.
2. **Preprocessing** (`src/mm_preprocess.py`) — prevalence/abundance filtering,
   CLR (centered-log-ratio) transform, clinical variable selection/imputation,
   and integer-count export for DESeq2.
3. **Per-modality analysis**
   - *Metagenomics* — alpha/beta diversity + PERMANOVA, CLR linear-model
     differential abundance, **DESeq2** count-based differential abundance with
     **pathogen annotation**.
   - *Functional pathways* — PCA ordination, CLR differential testing.
   - *Clinical* — group tests, distributions, score clustering, and an
     **L1-regularized logistic regression** predicting group.
4. **Multi-omic integration**
   - **MOFA2** (unsupervised factors; factor–clinical correlation; **XGBoost**
     on factors → group),
   - **DIABLO** (supervised, `mixOmics`),
   - **SNF** (similarity network fusion) + Mantel/Procrustes/PLS concordance,
   - **within-autism sPLS** (omics ↔ ASD symptom scores),
   - **differential co-abundance network** (autism vs control).
5. **Report** (`src/mm_report.py`) — Nature-style multi-panel
   `data/multiomics_report.pdf`.

All differential results are age/sex-adjusted and FDR-controlled. Every stage
writes a log to `results/logs/`.

---

## Repository layout

```
src/
  harmonization.py                 # step 0: build aligned data/*.tsv from raw inputs
  mm_common.py                     # shared utilities, paths, colors, CLR, stats
  mm_preprocess.py                 # filtering, CLR, clinical selection, counts
  mm_modality_metagenomics.py      # diversity, PERMANOVA, CLR differential abundance
  mm_deseq2.R                      # DESeq2 count-based differential abundance
  mm_pathogens.py                  # annotate DESeq2 hits vs curated pathogen list
  mm_modality_metabolomics.py      # functional-pathway PCA + differential testing
  mm_modality_clinical.py          # clinical tests + L1 logistic regression
  mm_integration_mofa.py           # MOFA2 + factor–clinical corr + XGBoost
  mm_integration_diablo.R          # DIABLO supervised integration (mixOmics)
  mm_integration_snf_concordance.py# SNF, concordance, within-autism sPLS, diff network
  mm_report.py                     # assemble the final multi-panel PDF report
  run_mm.sh                        # end-to-end orchestrator (steps 1–10)
  install_r_deps.R                 # installs mixOmics + DESeq2
requirements_mm.txt                # Python dependencies
PLAN.md, PLAN_MM.md                # design/methodology specifications
```

---

## Inputs (place under `data/`)

Used by `harmonization.py`:

| File | Description |
|---|---|
| `merged_bracken.tsv` | Bracken (Kraken 2) taxonomic table; `*.bracken_num` (counts) and `*.bracken_frac` (relative abundance) columns per sample |
| `metabolomics_raw_data/*_pathabundance.tsv` | HUMAnN per-sample pathway abundance files (one per sample) |
| `clinical.sav` | SPSS clinical records |
| `Sifrarnik_kakice.xlsx` | sample-ID mapping (Novogen ID ⇄ clinical code) |

Harmonization writes `data/metagenomics.tsv`, `data/metabolomics.tsv`,
`data/clinical.tsv` (consumed by the rest of the pipeline, which also reads
`merged_bracken.tsv` for DESeq2 counts).

---

## Setup

```bash
# Python (3.10+); a virtual environment is recommended
python3 -m venv .venv
.venv/bin/pip install -r requirements_mm.txt

# R (4.x) — installs mixOmics (DIABLO) and DESeq2 from Bioconductor
Rscript src/install_r_deps.R
```

---

## Run

```bash
# Step 0 — harmonize raw inputs into aligned data/*.tsv
.venv/bin/python src/harmonization.py

# Steps 1–10 — preprocessing, per-modality, integration, report
bash src/run_mm.sh
```

Final deliverable: **`data/multiomics_report.pdf`**. Intermediate figures
(`.png`) and tables (`.csv`) are written under `results/<stage>/`.

---

## Method notes & caveats

- **Compositional data**: taxonomic and functional abundances are CLR-transformed
  for all multivariate/differential analyses.
- **"Functional pathways", not metabolites**: the HUMAnN output is functional
  potential, labelled accordingly throughout.
- **Two differential-abundance views**: a conservative CLR linear model
  (age/sex-adjusted) and a more sensitive count-based DESeq2; both are reported.
- **Pathogen annotation** uses a **curated offline reference list** (not a live
  database query) and should be treated as a screen.
- **Clinical symptom scores** (SSDS/SDQ/GIRBI) exist for the autism group only;
  cross-group integration therefore uses two omics blocks, and the scores feed a
  within-autism sPLS instead.
- **Age** is derived as `2024 − birth year`; two samples with inconsistent age
  records are flagged.

## Dependencies

Python: pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn,
adjustText, scikit-bio, mofapy2, h5py, xgboost, pyreadstat, openpyxl.
R: mixOmics, DESeq2 (Bioconductor).
