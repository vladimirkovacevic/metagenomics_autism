#!/usr/bin/env Rscript
# Age-group-specific DIABLO integration (PLAN_MM.md §5.2, age-stratified).
# Fits a DIABLO (block sPLS-DA) model within each developmental age group
# (0-7, 8-12, 13+), reports cross-validated balanced error rate (BER) and AUC,
# and a label-permutation p-value for the multi-omic group separation.
# Outputs -> results/integration/diablo/  (per-stratum sample plots + perf CSV)

suppressMessages(library(mixOmics))
set.seed(1)

args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- normalizePath(file.path(if (length(fa)) dirname(fa) else getwd(), ".."))
PRE <- file.path(ROOT, "results", "preprocessed")
OUT <- file.path(ROOT, "results", "integration", "diablo")
logf <- file.path(ROOT, "results", "logs", "diablo_agegroups.log"); cat("", file = logf)
log <- function(...) { m <- sprintf(...); ts <- format(Sys.time(), "%H:%M:%S")
  cat(ts, "| diablo_age |", m, "\n"); cat(ts, "| diablo_age |", m, "\n", file = logf, append = TRUE) }

mg  <- read.delim(file.path(PRE, "metagenomics_clr.tsv"), row.names = 1, check.names = FALSE)
mb  <- read.delim(file.path(PRE, "metabolomics_clr.tsv"), row.names = 1, check.names = FALSE)
cov <- read.delim(file.path(PRE, "covariates.tsv"), row.names = 1, check.names = FALSE)
ages <- c("0-7", "8-12", "13+")
grp_col <- c("#3287be", "#ff7d55")
NPERM <- 199

# observed multi-omic association statistic = in-sample consensus AUC (ncomp=1),
# compared against the same statistic under NPERM permutations of the labels.
insample_auc <- function(X, Y) {
  m <- block.splsda(X, Y, ncomp = 1, keepX = lapply(X, function(z) 10), design = "full")
  a <- auroc(m, plot = FALSE, print = FALSE)
  vals <- unlist(lapply(a, function(b) b[[1]][1]))   # per-block AUC, comp 1
  mean(vals, na.rm = TRUE)
}

perf_rows <- list()
for (k in seq_along(ages)) {
  agl <- ages[k]
  idx <- rownames(cov)[!is.na(cov$age_group) & cov$age_group == agl]
  idx <- intersect(idx, rownames(mg))
  Y <- factor(cov[idx, "group"])
  if (nlevels(Y) < 2 || length(idx) < 12) { log("skip %s (n=%d)", agl, length(idx)); next }
  X <- list(metagenomics = as.matrix(mg[idx, ]), pathways = as.matrix(mb[idx, ]))
  keepX <- list(metagenomics = c(10, 10), pathways = c(10, 10))
  model <- block.splsda(X, Y, ncomp = 2, keepX = keepX, design = "full")

  # cross-validated performance
  pf <- perf(model, validation = "Mfold", folds = 5, nrepeat = 5, auc = TRUE)
  ber <- tryCatch(pf$WeightedVote.error.rate$centroids.dist["Overall.BER", 2], error = function(e) NA)
  auc_cv <- tryCatch(mean(sapply(names(X), function(b) {
    a <- pf$auc[[b]]; a[[length(a)]][1] })), error = function(e) NA)

  # permutation test on the in-sample multi-omic AUC
  obs <- insample_auc(X, Y)
  perm <- replicate(NPERM, insample_auc(X, factor(sample(as.character(Y)))))
  pval <- (1 + sum(perm >= obs)) / (NPERM + 1)

  perf_rows[[agl]] <- data.frame(age_group = agl, n = length(idx),
                                 BER = round(ber, 3), AUC_cv = round(auc_cv, 3),
                                 AUC_insample = round(obs, 3), perm_p = round(pval, 4))
  log("%s: n=%d BER=%.3f AUC_cv=%.3f perm_p=%.4f", agl, length(idx), ber, auc_cv, pval)

  png(file.path(OUT, sprintf("diablo_age_%d.png", k)), width = 1400, height = 700, res = 170)
  tryCatch(print(plotIndiv(model, ind.names = FALSE, legend = TRUE, ellipse = TRUE,
                           col.per.group = grp_col,
                           title = sprintf("DIABLO — age %s (n=%d)", agl, length(idx)))),
           error = function(e) plot.new())
  dev.off()
}
res <- do.call(rbind, perf_rows)
write.csv(res, file.path(OUT, "diablo_agegroup_perf.csv"), row.names = FALSE)
log("age-group DIABLO complete -> results/integration/diablo")
