#!/usr/bin/env bash
# End-to-end multi-omics analysis pipeline (PLAN_MM.md).
# Runs preprocessing -> per-modality analyses -> integration -> final PDF report.
# Prereqs: .venv with requirements_mm.txt; R with mixOmics (src/install_r_deps.R).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[1/10] preprocess";                  $PY src/mm_preprocess.py
echo "[2/10] metagenomics";                $PY src/mm_modality_metagenomics.py
echo "[3/10] DESeq2 (R)";                   Rscript src/mm_deseq2.R
echo "[4/10] pathogen annotation";          $PY src/mm_pathogens.py
echo "[5/10] metabolomics (functional)";   $PY src/mm_modality_metabolomics.py
echo "[6/10] clinical";                     $PY src/mm_modality_clinical.py
echo "[7/10] MOFA2 + XGBoost";              $PY src/mm_integration_mofa.py
echo "[8/10] DIABLO (R)";                   Rscript src/mm_integration_diablo.R
echo "[9/10] SNF + concordance + netdiff";  $PY src/mm_integration_snf_concordance.py
echo "[10/10] final PDF report";            $PY src/mm_report.py
echo "Done -> data/multiomics_report.pdf"
