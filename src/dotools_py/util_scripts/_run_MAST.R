#######################################################################################
# Description:  Running MAST test for scRNA-seq from AnnData                          #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 08-06-2025							                                  #
# Date Modified: 08-06-2025                                                           #
# Version: 1.0									                                      #
# R Version: 4.3.2               						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(SingleCellExperiment)))
suppressWarnings(suppressMessages(library(Biobase)))
suppressWarnings(suppressMessages(library(MAST)))


option_list <- list(
    make_option(c("--input"), type = "character", default = NULL,
                help = "Absolute path to Anndata object (H5AD File)", metavar = "character"),
    make_option(c("--out"), type = "character", default = NULL,
                help = "Absolute path to the directory where the output files will be saved",
                metavar = "character"),
    make_option(c("--key"), type = "character", default = NULL,
                help = "Column name in .obs attribute in the anndata that contains the condition information",
                metavar = "character"),
    make_option(c("--ref"), type = "character", default = NULL,
                help = "Name of reference condition",
                metavar = "character"),
    make_option(c("--disease"), type = "character", default = NULL,
                help = "Name of alternative condition",
                metavar = "character"),
    make_option(c("--covariates"), type = "character", default = NULL,
                help = "Extra column with covariates to consider",
                metavar = "character")
)

opt_parser <- OptionParser(usage = "usage: %prog [options]

Perform differential gene expression using the MAST test on an Anndata object and output a
CSV file with the differentially expressed genes.",
option_list = option_list)

opt <- parse_args(opt_parser)

if (is.null(opt$input)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$out)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$key)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}else if (is.null(opt$ref)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}else if (is.null(opt$disease)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}


message('Reading AnnData in R')
sce <- zellkonverter::readH5AD(opt$input)

# Re-organise to set reference as first level
condition <- factor(x = SummarizedExperiment::colData(sce)[[opt$key]])
condition <- relevel(x = condition, ref = opt$ref)
print_ref <- levels(condition)
message("Reference condition has been set: ", print_ref[[1]])
sce$condition <- condition
latent.vars <- c('condition', opt$covariates)

# Generate SingleCellAssay for MAST
sca <- FromMatrix(exprsArray =  as.matrix(logcounts(sce)),
                  cData =  as.data.frame(colData(sce)),
                  fData = as.data.frame(rowData(sce))
)

formula <- as.formula(object = paste0(" ~ ", paste(latent.vars, collapse = "+")))


# Run Test
message('Running MAST Test')
zlmCond <- MAST::zlm(formula = formula, sca = sca)
summaryCond <- MAST::summary(object = zlmCond, doLRT = paste0(opt$key, opt$disease))
summaryDt <- summaryCond$datatable

# Generate Results dataframe
p_val <- as.data.frame(summaryDt[summaryDt[["component"]] == "H", 'Pr(>Chisq)'])
names(p_val) <- 'pvals'
genes <- as.data.frame(summaryDt[summaryDt[["component"]] == "H", 1])
p_val[['padj']] <- p.adjust(p_val$pvals, method = 'fdr')


#log1pdata.mean.fxn <- function(x) {return(log(x = (rowSums(x = expm1(x = x)) + 1)/NCOL(x), base = 2))}
#counts_mat <- as.matrix(counts(sce))  # convert to dense matrix if sparse
#logcounts_mat <- as.matrix(logcounts(sce))
#idx1 <- which(condition == opt$ref)
#idx2 <- which(condition == opt$disease)
#mean1 <- log1pdata.mean.fxn(logcounts_mat[,idx1])
#mean2 <-  log1pdata.mean.fxn(logcounts_mat[,idx2])
#log2fc <- as.data.frame(mean2 - mean1)
#pct1 <- rowMeans(counts_mat[, idx1, drop=FALSE] > 0)
#pct2 <- rowMeans(counts_mat[, idx2, drop=FALSE] > 0)

df <-data.frame(
    names = genes,
    pvals = p_val$pvals,
    #log2c = log2fc,
    padj = p_val$padj
    #pts_ref = pct1,
    #pts_group = pct2
  )

message('Saving DGE Table')
write.csv(df, opt$out)
