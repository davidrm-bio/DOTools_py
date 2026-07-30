#######################################################################################
# Description:  Convert RDS Object (SCE/Seurat) to AnnData                            #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 30-07-2026							                                  #
# Date Modified: 30-07-2026                                                           #
# Version: 2.0									                                      #
# R Version: 4.6.1 (Seurat 5.5.1)						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(anndataR)))
suppressWarnings(suppressMessages(library(Seurat)))


option_list <- list(
    make_option("--input", type = "character", default = NULL,
                help = "Absolute path to RDS object", metavar = "character"),
    make_option("--out", type = "character", default = NULL,
                help = "Absolute path to the directory where the output files will be saved",
                metavar = "character"),
    make_option("--type", type = "character", default = 'SeuratObject',
                help = "Type of Object to convert to (SingleCellExperiment or SeuratObject)",
                metavar = "character"),
    make_option("--operation", type = "character", default = NULL,
                help = "Type of convertion: read (RDS --> AnnData) or write (AnnData --> RDS)",
                metavar = "character"),
    make_option("--batch_key", type = "character", default = "batch",
                help = "Batch key in AnnData", metavar = "character")
)


opt_parser <- OptionParser(usage = "usage: %prog [options]
Convertion between SingleCellExperiment, Seurat and AnnData
Objects.", option_list = option_list)

opt <- parse_args(opt_parser)

if (is.null(opt$input)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$out)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$type)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
} else if (is.null(opt$operation)) {
    print_help(opt_parser)
    stop("Please provide the specified arguments", call. = FALSE)
}



if (opt$operation == 'read') {  # Convert RDS (SCE/Seurat) to AnnData
    message("Converting RDS Object to AnnData Object")
    input.obj <- readRDS(opt$input) # Read SCE/Seurat

    output.obj <- as_AnnData(input.obj) # Convert to AnnData

  # Transfer missing information if we come from SeuratObject
    if (is(input.obj, "Seurat")) {

        hvg <- VariableFeatures(input.obj)
        if (length(hvg) == 0) {
            message("No HVGs found in the RDS Object")
        }

        graph_names <- sub("RNA_", "", names(input.obj@graphs))
        for (name in graph_names) {
          if (!name %in% output.obj$obsp_keys()) {
            message(paste0("The graph RNA_", name, " could not be transferred"))
          }
        }

        if (length(output.obj$obsm_keys()) != length(input.obj@reductions)){
          reduction_names <- Reductions(input.obj)
          for (name in reduction_names) {
            output.obj$obsm[[paste0("X_", name)]] <- input.obj[[name]]@cell.embeddings
          }
        }
    }
    suppressWarnings(suppressMessages(output.obj$write_h5ad(opt$out)))

} else if (opt$operation == 'write') {  # Convert AnnData to RDS
    message("Converting AnnData Object to RDS Object")

    output.obj <- read_h5ad(opt$input, as = opt$type) # Read AnnData Object

    # Transfer missing information for SeuratObject
    if (opt$type == "Seurat") {
        input.obj <- read_h5ad(opt$input)
        DefaultAssay(output.obj) <- "RNA"

        # Replace orig.ident with batch_key
        message("Saving batch information")
        output.obj <- tryCatch({
            output.obj$orig.ident <- output.obj@meta.data[opt$batch_key]
            output.obj
        }, error = function(e) {
            message("Error while renaming batch_key: ", e$message)
            return(output.obj) }
        )

        # Update nCounts_RNA and nFeatures_RNS
        if ("total_counts" %in% colnames(output.obj@meta.data)) {
            output.obj$nCount_RNA <- output.obj$total_counts
        }
        if ("n_genes" %in% colnames(output.obj@meta.data)) {
            output.obj$nFeature_RNA <- output.obj$n_genes
        }

        # Transfer other missing elements
        tmp_folder <- strsplit(opt$input, "/")[[1]]
        tmp_folder <- tmp_folder[-length(tmp_folder)]
        tmp_folder <- paste(tmp_folder, collapse = "/")

        # VariableFeatures
        message("Getting highly variable genes")
        hvg <- tryCatch({
            hvg <- input.obj$var_names[input.obj$var$highly_variable]
        }, error = function(e) {
            message("Skipping\nError while transfering HVGs: ", e$message)
            return(NULL) })

        if (!is.null(hvg)) {
          VariableFeatures(output.obj[["RNA"]]) <- hvg
        }

        # Rename reductions to remove X_ and make lowercase
        message("Renaming reduction assays")
        reductions_names <- names(output.obj@reductions)
        reductions_names_clean <- tolower(sub("^X_", "", reductions_names))

        for (i in 1:length(reductions_names)) {
            output.obj@reductions[[reductions_names_clean[i]]] <- output.obj@reductions[[reductions_names[i]]]
            output.obj@reductions[[reductions_names[i]]] <- NULL
            output.obj@reductions[[reductions_names_clean[i]]]@assay.used <- "RNA"
        }

        # Connectivities -> snn
        message("Updating Graph")
        graphs_names <- names(output.obj@graphs)
        graphs_names <- ifelse(
          graphs_names == "RNA_connectivities", "RNA_snn",
          ifelse(
            graphs_names == "RNA_distances", "RNA_nn",
            ifelse(
              grepl("_connectivities$", graphs_names), "RNA_snn",
              ifelse(
                grepl("_distances$", graphs_names), "RNA_nn", graphs_names)
              )
            )
        )
        names(output.obj@graphs) <- graphs_names

        message("Updating layer names to match Seurat requirements")
        layers <- Layers(output.obj[["RNA"]])

        if (!"data" %in% layers && "logcounts" %in% layers) {
            output.obj[["RNA"]]$data <- LayerData(
                output.obj[["RNA"]],
                layer = "logcounts"
            )
        }
        if ("logcounts" %in% Layers(output.obj[["RNA"]])) {
            output.obj[["RNA"]]$logcounts <- NULL
        }
        if ("X" %in% Layers(output.obj[["RNA"]])) {
            output.obj[["RNA"]]$X <- NULL
        }

    }
    saveRDS(output.obj, opt$out)
} else {
    stop("Only read and write operations are permitted")
}
