#!/usr/bin/env Rscript
# DIABLO supervised multi-omics integration (PLAN_MM.md §5.2).
# Two-block model (metagenomics CLR + functional pathways CLR), Y = group.
# Produces performance (CV error + AUC), sample/arrow plots, loadings,
# circos plot, relevance network, and clustered image map (CIM).
# Outputs -> results/integration/diablo/.

suppressMessages(library(mixOmics))
set.seed(1)

args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
SCRIPT_DIR <- if (length(fa)) dirname(normalizePath(fa)) else getwd()
ROOT <- normalizePath(file.path(SCRIPT_DIR, ".."))
PRE  <- file.path(ROOT, "results", "preprocessed")
OUT  <- file.path(ROOT, "results", "integration", "diablo")
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
logf <- file.path(ROOT, "results", "logs", "diablo.log")
log <- function(...) { msg <- sprintf(...); cat(format(Sys.time(), "%H:%M:%S"), "| diablo |", msg, "\n");
                       cat(format(Sys.time(), "%H:%M:%S"), "| diablo |", msg, "\n", file = logf, append = TRUE) }
cat("", file = logf)

# ---- load aligned data ---- #
mg  <- read.delim(file.path(PRE, "metagenomics_clr.tsv"), row.names = 1, check.names = FALSE)
mb  <- read.delim(file.path(PRE, "metabolomics_clr.tsv"), row.names = 1, check.names = FALSE)
cov <- read.delim(file.path(PRE, "covariates.tsv"), row.names = 1, check.names = FALSE)
stopifnot(all(rownames(mg) == rownames(mb)), all(rownames(mg) == rownames(cov)))
Y <- factor(cov$group)
log("DIABLO: %d samples | metagenomics %d feats | pathways %d feats | groups %s",
    nrow(mg), ncol(mg), ncol(mb), paste(levels(Y), collapse = "/"))

X <- list(metagenomics = as.matrix(mg), pathways = as.matrix(mb))
design <- matrix(0.5, ncol = length(X), nrow = length(X),
                 dimnames = list(names(X), names(X)))
diag(design) <- 0

ncomp <- 2
keepX <- list(metagenomics = c(15, 15), pathways = c(15, 15))  # sparse selection per comp
model <- block.splsda(X, Y, ncomp = ncomp, keepX = keepX, design = design)
log("model fitted: ncomp=%d, keepX=15/comp/block, design off-diag=0.5", ncomp)

# ---- performance: M-fold CV error + AUC ---- #
set.seed(1)
perf.res <- perf(model, validation = "Mfold", folds = 5, nrepeat = 10, auc = TRUE)
err <- perf.res$WeightedVote.error.rate$centroids.dist
write.csv(err, file.path(OUT, "performance_error_rate.csv"))
log("CV weighted-vote error rate (centroids, per comp): %s",
    paste(round(err["Overall.ER", ], 3), collapse = ", "))
# AUC per block (last component)
auc_block <- sapply(names(X), function(b) {
  a <- perf.res$auc[[b]]
  if (is.null(a)) NA else a[[length(a)]][1]
})
write.csv(data.frame(block = names(auc_block), AUC = auc_block),
          file.path(OUT, "performance_auc.csv"), row.names = FALSE)

# ---- selected variables (loadings) per block per component ---- #
for (b in names(X)) {
  for (cc in 1:ncomp) {
    sv <- selectVar(model, block = b, comp = cc)[[b]]
    if (!is.null(sv$value)) {
      df <- data.frame(feature = rownames(sv$value), loading = sv$value[, 1])
      write.csv(df, file.path(OUT, sprintf("loadings_%s_comp%d.csv", b, cc)), row.names = FALSE)
    }
  }
}

# `expr` is forced inside print() so ggplot-returning calls (plotIndiv/plotArrow)
# actually draw to the open device; base-graphics calls print invisibly.
savep <- function(name, w = 1500, h = 1300, expr) {
  fp <- file.path(OUT, name)
  tryCatch({
    png(fp, width = w, height = h, res = 170); print(expr); dev.off()
    if (file.exists(fp)) log("wrote %s", name) else log("SKIP %s: no file produced", name)
  }, error = function(e) { if (dev.cur() > 1) dev.off(); log("SKIP %s: %s", name, conditionMessage(e)) })
}

grp_col <- c("#3287be", "#ff7d55")  # autism, control (PLAN_MM.md §6 scheme)
savep("sample_plot.png", expr =
  plotIndiv(model, ind.names = FALSE, legend = TRUE, ellipse = TRUE,
            col.per.group = grp_col,
            title = "DIABLO sample plot (autism vs control)"))
savep("arrow_plot.png", expr =
  plotArrow(model, ind.names = FALSE, legend = TRUE, col.per.group = grp_col,
            title = "DIABLO arrow plot (block consensus)"))
savep("loadings_comp1.png", w = 2400, h = 1400, expr =
  plotLoadings(model, comp = 1, contrib = "max", method = "median",
               size.name = 0.6, size.title = 1, title = "DIABLO loadings (comp 1)"))
savep("circos.png", w = 1600, h = 1600, expr =
  circosPlot(model, cutoff = 0.5, size.variables = 0.5, line = TRUE,
             title = "DIABLO circos (|corr|>0.5)"))
savep("cim.png", w = 1600, h = 1500, expr =
  cimDiablo(model, margins = c(10, 14), size.legend = 0.6))
savep("network.png", w = 1600, h = 1400, expr = {
  net <- network(model, cutoff = 0.5, save = NULL, plot.graph = TRUE)
})

log("DIABLO analysis complete -> %s", file.path("results", "integration", "diablo"))
