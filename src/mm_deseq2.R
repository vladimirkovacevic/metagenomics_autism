#!/usr/bin/env Rscript
# DESeq2 differential abundance for metagenomic taxon counts (PLAN_MM.md §4.1).
# Negative-binomial model on Bracken read counts, adjusted for age + sex.
# Output -> results/metagenomics/deseq2_results.csv (+ volcano PNG).

suppressMessages({library(DESeq2)})
set.seed(1)

args <- commandArgs(trailingOnly = FALSE)
fa <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- normalizePath(file.path(if (length(fa)) dirname(fa) else getwd(), ".."))
PRE <- file.path(ROOT, "results", "preprocessed")
OUT <- file.path(ROOT, "results", "metagenomics")
logf <- file.path(ROOT, "results", "logs", "deseq2.log"); cat("", file = logf)
log <- function(...) { m <- sprintf(...); ts <- format(Sys.time(), "%H:%M:%S")
  cat(ts, "| deseq2 |", m, "\n"); cat(ts, "| deseq2 |", m, "\n", file = logf, append = TRUE) }

counts <- as.matrix(read.delim(file.path(PRE, "metagenomics_counts.tsv"),
                               row.names = 1, check.names = FALSE))
cov <- read.delim(file.path(PRE, "covariates.tsv"), row.names = 1, check.names = FALSE)
cov <- cov[colnames(counts), ]
cov$age <- as.numeric(cov$age)

# complete cases for the design (drop the 2 samples missing age/sex)
keep_s <- !is.na(cov$age) & !is.na(cov$sex) & !is.na(cov$group)
counts <- counts[, keep_s]; cov <- cov[keep_s, ]
cov$group <- relevel(factor(cov$group), ref = "control")   # +log2FC = up in autism
cov$sex <- factor(cov$sex)

# prefilter very low-count taxa
counts <- counts[rowSums(counts) >= 10, ]
log("DESeq2 input: %d taxa x %d samples (%s)", nrow(counts), ncol(counts),
    paste(levels(cov$group), collapse = "/"))

dds <- DESeqDataSetFromMatrix(countData = counts, colData = cov,
                              design = ~ age + sex + group)
dds <- DESeq(dds, quiet = TRUE)
res <- results(dds, name = "group_autism_vs_control")
res <- as.data.frame(res[order(res$padj), ])
res$taxon <- rownames(res)
res <- res[, c("taxon", "baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj")]
write.csv(res, file.path(OUT, "deseq2_results.csv"), row.names = FALSE)
nsig <- sum(res$padj < 0.05, na.rm = TRUE)
log("DESeq2 complete: %d taxa significant at padj<0.05", nsig)

# volcano
png(file.path(OUT, "deseq2_volcano.png"), width = 1500, height = 1200, res = 200)
sigcol <- ifelse(is.na(res$padj) | res$padj >= 0.05, "#B0B0B0",
                 ifelse(res$log2FoldChange > 0, "#d7191c", "#2c7bb6"))
plot(res$log2FoldChange, -log10(res$padj), pch = 19, cex = 0.6, col = sigcol,
     xlab = "log2 fold change (autism vs control)", ylab = "-log10(padj)",
     main = sprintf("DESeq2 differential taxa (%d sig at padj<0.05)", nsig))
abline(h = -log10(0.05), lty = 2)
dev.off()
log("wrote deseq2_volcano.png")
