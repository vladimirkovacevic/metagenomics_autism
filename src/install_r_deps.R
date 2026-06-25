#!/usr/bin/env Rscript
# R dependencies for the pipeline. Run once:  Rscript src/install_r_deps.R
#   mixOmics  -> DIABLO supervised integration (mm_integration_diablo.R)
#   DESeq2    -> count-based differential abundance (mm_deseq2.R)
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
for (pkg in c("mixOmics", "DESeq2")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    BiocManager::install(pkg, update = FALSE, ask = FALSE)
  cat(pkg, "installed:", requireNamespace(pkg, quietly = TRUE), "\n")
}
