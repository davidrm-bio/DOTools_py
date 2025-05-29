#######################################################################################
# Description:  Find doublets using scDblFinder                                       #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 03-04-2025							                                  #
# Date Modified: 03-04-2025                                                           #
# Version: 1.0									                                      #
# R Version: 4.4.1 (Seurat 5.2.1)						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(scDblFinder)))
suppressWarnings(suppressMessages(library(data.table)))


option_list <- list(
  make_option(c("--input"), type="character", default=NULL,
              help="Absolute path to Anndata object (H5AD File)", metavar="character"),
  make_option(c("--out"), type="character", default=NULL,
              help="Absolute path to the directory where the output files will be saved",
              metavar="character"),
  make_option(c("--name"), type="character", default=NULL,
              help="Column name in .obs attribute in the anndata that contains the batch information",
              metavar="character")
)

opt_parser <- OptionParser(usage = "usage: %prog [options]
Find doublets in an Anndata object with one or multiple samples and output a
CSV file with the doublet score and class. You must specify the colname with batch
information when running the inference in an Anndata object with multiple samples.",
option_list=option_list)

opt <- parse_args(opt_parser)

if (is.null(opt$input)){
  print_help(opt_parser)
  stop("Please provide the specified arguments", call.=FALSE)
  } else if (is.null(opt$out)) {
  print_help(opt_parser)
  stop("Please provide the specified arguments", call.=FALSE)
}  # name argument is optional


# Read h5ad as ScE
message('Reading AnnData in R')
sce <- readH5AD(opt$input)

# Find doublets
if (!is.null(opt$name)) {
  message('Finding doublets per batch...')
  sce <- scDblFinder(sce, samples = opt$name, verbose = F)
} else {
    message('Finding doublets...')
  sce <- scDblFinder(sce, verbose = F)

}

df <- as.data.frame(colData(sce)[c('scDblFinder.class', 'scDblFinder.score')])

# Save csv with metadata information
message('Exporting to Python')
fwrite(df, paste0(opt$out, 'scDblFinder_inference.csv'))
