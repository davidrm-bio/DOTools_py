#######################################################################################
# Description:  Seurat Anchor Integration on Anndata (Incorporate in Scanpy Workflow) #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 16-02-2024							                                  #
# Date Modified: 02-08-2024                                                           #
# Version: 1.0									                                      #
# R Version: 4.3.2 (Seurat 5.0.1)						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
suppressWarnings(suppressMessages(library(Seurat)))
suppressWarnings(suppressMessages(library(data.table)))


option_list <- list(
    make_option(c("--input"), type = "character", default = NULL,
                help = "Absolute path to Anndata object (H5AD File)", metavar = "character"),
    make_option(c("--out"), type = "character", default = NULL,
                help = "Absolute path to the directory where the output files will be saved",
                metavar = "character"),
    make_option(c("--name"), type = "character", default = NULL,
                help = "Column name in .obs attribute in the anndata that contains the batch information",
                metavar = "character"),
    make_option(c("--version"), type = "character", default = "v4",
                help = "Seurat version approach to use (v4 or v5)",
                metavar = "character")
)

opt_parser <- OptionParser(usage = "usage: %prog [options]

Perform Seurat Anchor integration on an Anndata object and output a
CSV file with the batch corrected matrix. You should provide the path
to an anndata object that only contains the highly variable genes.",
                           option_list = option_list)

opt <- parse_args(opt_parser)

if (is.null(opt$input)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$out)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$name)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}


# Read h5ad as ScE
message('Reading AnnData in R')
adata.hvg <- readH5AD(opt$input)
Seu.hvg <- as.Seurat(adata.hvg, counts = "counts", data = 'logcounts')
# DefaultAssay(Seu.hvg) <- "originalexp"

# Anchor Integration
message('Starting ACC in R environment...')
if (opt$version == 'v4') {
    message("Run CCA with v4")
    batch.list <- SplitObject(Seu.hvg, split.by = opt$name)
    anchors <- FindIntegrationAnchors(object.list = batch.list, anchor.features = rownames(Seu.hvg))
    integrated <- IntegrateData(anchors)
    integrated_expr <- GetAssayData(integrated)
    integrated_expr <- integrated_expr[rownames(Seu.hvg), colnames(Seu.hvg)]
    df_integrated_expr <- as.data.frame(integrated_expr)
    df_integrated_expr <- t(df_integrated_expr)

} else if (opt$version == 'v5') {
    message("Run CCA with v5")
    Seu.hvg[['originalexp']] <- split(Seu.hvg[['originalexp']], f = Seu.hvg@meta.data[[opt$name]])
    Seu.hvg <- ScaleData(Seu.hvg)
    Seu.hvg <- RunPCA(Seu.hvg, features = rownames(Seu.hvg), verbose=F)
    Seu.hvg <- IntegrateLayers(object = Seu.hvg, method = CCAIntegration, orig.reduction = "pca", new.reduction = "integrated.cca",
                               verbose = TRUE, assay="originalexp", features=rownames(Seu.hvg))
    CCAEmbeddings <- Seu.hvg@reductions$integrated.cca
    CCAEmbeddings <- CCAEmbeddings@cell.embeddings
    df_integrated_expr <- as.data.frame(CCAEmbeddings)
} else {
    stop("Specify v4 or v5 for the CCA approach to use")
}


# Prepare to export to python
message('Preparing to export to python...')
print(df_integrated_expr[1:3, 1:3])

message('Saving data in ')
message(paste0(opt$out, 'adata_hvg_seurat_AnchorIntegration.csv'))
fwrite(df_integrated_expr, paste0(opt$out, '/adata_hvg_seurat_AnchorIntegration.csv'))
