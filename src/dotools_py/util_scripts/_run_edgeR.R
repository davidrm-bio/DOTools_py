#######################################################################################
# Description:  Running edgeR test for scRNA-seq from AnnData                         #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 08-07-2025							                                  #
# Date Modified: 08-07-2025                                                           #
# Version: 1.0									                                      #
# R Version: 4.3.2               						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(SingleCellExperiment)))
suppressWarnings(suppressMessages(library(edgeR)))


option_list <- list(
    make_option("--input", type = "character", default = NULL,
                help = "Absolute path to Anndata object (H5AD File)",
                metavar = "character"),
    make_option("--out", type = "character", default = NULL,
                help = "Absolute path to the directory where the output files will be saved",
                metavar = "character"),
    make_option("--batch", type = "character", default = NULL,
                help = "Column name in .obs attribute in the anndata that contains the batch information",
                metavar = "character"),
    make_option("--condition", type = "character", default = NULL,
                help = "Column name in .obs attribute in the anndata that contains the condition information",
                metavar = "character"),
    make_option("--ref", type = "character", default = NULL,
                help = "Name of the reference condition",
                metavar = "character"),
    make_option("--disease", type = "character", default = NULL,
                help = "Name of the alternative condition",
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
} else if (is.null(opt$batch)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}else if (is.null(opt$condition)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}else if (is.null(opt$ref)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}else if (is.null(opt$disease)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}

fit_model <- function(adata_) {
    # Create edgeR object
    y <- edgeR::DGEList(assay(adata_, "X"), group = colData(adata_)[[opt$batch]])

    # Filter lowly expressed genes
    keep <- edgeR::filterByExpr(y)
    y <- y[keep, , keep.lib.sizes=FALSE]

    # Normalize
    y <- edgeR::calcNormFactors(y)

    # Define design
    group <- colData(adata_)[[opt$condition]]
    design <- model.matrix(~ 0 + group)

    # Estimate dispersion and fit
    y <- edgeR::estimateDisp(y, design)
    fit <- edgeR::glmQLFit(y, design)

    return(list("fit" = fit, "design" = design, "y" = y))
}

message('Reading AnnData in R')
sce <- zellkonverter::readH5AD(opt$input)

message('Running edgeR Test')
outs <-fit_model(sce)
fit <- outs$fit
y <- outs$y

message('Generating DGE Table to pass to Python')
contrast_name <- paste0('group', opt$disease, '-', 'group', opt$ref)
myContrast <- makeContrasts(contrast_name, levels = y$design)
qlf <- glmQLFTest(fit, contrast=myContrast)
degs <- topTags(qlf, n = Inf)
degs <- degs$table
write.csv(degs, opt$out)
