#######################################################################################
# Description:  Convert RDS Object (SCE/Seurat) to AnnData                            #
#										                                              #
# Author: David Rodriguez Morales						                              #
# Date Created: 16-07-2025							                                  #
# Date Modified: 16-07-2025                                                           #
# Version: 1.0									                                      #
# R Version: 4.3.2 (Seurat 5.3.0)						                              #
#######################################################################################

suppressWarnings(suppressMessages(library(optparse)))
suppressWarnings(suppressMessages(library(zellkonverter)))
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

TransformObjectType <- function(obj, type) {
    if (is(obj, "Seurat")) {
        obj_type <- "SeuratObject"
    } else if (is(obj, "SingleCellExperiment")) {
        obj_type <- "SingleCellExperiment"
    }

    if (obj_type == type) {
        return(obj)  # The class we want is the same
    } else if (obj_type == "SingleCellExperiment" && type == "SeuratObject") {
        seu.obj <- as.Seurat(obj, counts = "counts", data = "logcounts")  # We have SCE and want SeuratObject
        return(seu.obj)
    } else if (obj_type == "SeuratObject" && type == "SingleCellExperiment") {
        sce <- Seurat::as.SingleCellExperiment(obj)  #We have SeuratObject and want SingleCellExperiment
        return(sce)
    }
}


if (opt$operation == 'read') {  # Convert RDS (SCE/Seurat) to AnnData
    message("Converting RDS Object to AnnData Object")
    input.obj <- readRDS(opt$input)
    output.obj <- TransformObjectType(input.obj, "SingleCellExperiment")
    writeH5AD(output.obj, opt$out)  # SCE does not save HVGs and Graphs

    # Transfer missing information if we come from SeuratObject
    if (is(input.obj, "Seurat")) {
        tmp_folder <- strsplit(opt$input, "/")[[1]]
        tmp_folder <- tmp_folder[-length(tmp_folder)]
        tmp_folder <- paste(tmp_folder, collapse = "/")

        # Get all the available assays
        assay_names <- names(input.obj@assays)
        graph_names <- names(input.obj@graphs)

        # Initialise vectors to save HVGs and Graphs
        hvg <- c()
        snn <- c()
        nn <- c()
        for (current_assay in assay_names) {
            DefaultAssay(input.obj) <- current_assay

            # Get HVGs
            if (length(hvg) == 0) { hvg <- VariableFeatures(input.obj) }

            # Get Graphs
            for (assay in graph_names) {
                if (length(snn) == 0) {
                    if (grepl("_snn", assay)) {
                        snn <- as.matrix(input.obj@graphs[[assay]])
                    }
                }
                if (length(nn) == 0) {
                    if (grepl("_nn", assay)) {
                        nn <- as.matrix(input.obj@graphs[[assay]])
                    }
                }
            }
        }

        if (length(hvg) == 0) {
            message("No HVGs found in the object")
        } else {
            write.csv(as.data.frame(hvg), paste0(tmp_folder, "/VariableFeatures.csv"))
        }

        if (length(snn) == 0) {
            message("No SNN Graph found in the object")
        } else {
            data.table::fwrite(snn, paste0(tmp_folder, "/Connectivities.csv"))
        }

        if (length(nn) == 0) {
            message("No NN Graph found in the object")
        } else {
            data.table::fwrite(nn, paste0(tmp_folder, "/Distances.csv"))
        }
    }

} else if (opt$operation == 'write') {  # Convert AnnData to RDS
    message("Converting AnnData Object to RDS Object")
    input.obj <- readH5AD(opt$input)
    output.obj <- TransformObjectType(input.obj, opt$type)

    # Transfer missing information for SeuratObject
    if (opt$type == "SeuratObject") {

        # Rename Assay to RNA
        message("Generating RNA assay")
        assay_name <- names(output.obj@assays)  # Should contain originalexp only
        if (length(assay_name) > 1) {
            if (grepl("originalexp", assay_name)) {
                assay_name <- "originalexp"
            }
        }
        output.obj[["RNA"]] <- output.obj[[assay_name]]
        DefaultAssay(output.obj) <- "RNA"
        output.obj[[assay_name]] <- NULL

        # Replace orig.ident with batch_key
        message("Saving batch information")
        output.obj <- tryCatch({
            output.obj$orig.ident <- output.obj@meta.data[opt$batch_key]
        }, error = function(e) {
            message("Error while renaming batch_key: ", e$message)
            return(output.obj) }
        )

        #output.obj$orig.ident <- output.obj@meta.data[opt$batch_key]

        # Remove nCount_originalexp, nFeature_originalexp
        output.obj$nCount_originalexp <- NULL
        output.obj$nFeature_originalexp <- NULL
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
            hvg <- read.csv(paste0(tmp_folder, "/VariableFeatures.csv"))
            hvg <- hvg[hvg$highly_variable == "True", "X"]
        }, error = function(e) {
            message("Error while transfering HVGs: ", e$message)
            return(NULL) })

        if (!is.null(hvg)) {
            output.obj@assays$RNA@var.features <- hvg
        }

        # Rename reductions to remove X_ and make lowercase
        message("Renaming reduction assays")
        reductions_names <- names(output.obj@reductions)
        reductions_names_clean <- tolower(sub("^X_", "", reductions_names))

        for (i in 1:length(reductions_names)) {
            output.obj@reductions[reductions_names_clean[[i]]] <- output.obj@reductions[reductions_names[[i]]]
            output.obj@reductions[reductions_names[[i]]] <- NULL
            output.obj@reductions[reductions_names[[i]]]@assay.used <- "RNA"
        }

        # Connectivities -> snn
        message("Getting SNN Graph")
        snn <- tryCatch(
        {
            # Try to read the file
            connectivities <- data.table::fread(paste0(tmp_folder, "/Connectivities.csv"),
                                                check.names = FALSE, stringsAsFactors = FALSE)
            connectivities <- as.matrix(connectivities)

            if ("V1" %in% colnames(connectivities)) {
                connectivities$V1 <- NULL
            }
            rownames(connectivities) <- colnames(connectivities)
            connectivities_sparse <- as(connectivities, "dgCMatrix")
            snn <- as.Graph(x = connectivities_sparse)
            slot(snn, name = "assay.used") <- "RNA"

            snn

        }, error = function(e) {
                # It can fail due to memory limit
                msg <- conditionMessage(e)
                if (grepl("vector memory limit of ", msg)) {  # Try increase memory limit
                    options(mem.maxVSize = 128e9)  # 128 GB
                    tryCatch({
                        connectivities <- data.table::fread(paste0(tmp_folder, "/Connectivities.csv"),
                                                            check.names = FALSE, stringsAsFactors = FALSE)
                        connectivities <- as.matrix(connectivities)

                        if ("V1" %in% colnames(connectivities)) {
                            connectivities$V1 <- NULL
                        }
                        rownames(connectivities) <- colnames(connectivities)
                        connectivities_sparse <- as(connectivities, "dgCMatrix")
                        snn <- as.Graph(x = connectivities_sparse)
                        slot(snn, name = "assay.used") <- "RNA"
                        snn
                    },
                        error = function(e2) {  # If the second time does not work return NULL
                            message("Error while transfering SNN Graph: ", e$message)
                            return(NULL)
                        })
                } else {  # If it is another type of error return
                    message("Error while transfering SNN Graph: ", e$message)
                    return(NULL)
                }
            }
        )

        if (!is.null(snn)) {
            output.obj@graphs$RNA_snn <- snn
            output.obj@graphs$RNA_snn@assay.used <- "RNA"

        }

        # distances -> nn
        message("Getting NN Graph")
        nn <- tryCatch({
            distances <- data.table::fread(paste0(tmp_folder, "/Distances.csv"), check.names = FALSE,
                                           stringsAsFactors = FALSE)
            if ("V1" %in% colnames(distances)) {
                distances$V1 <- NULL
            }
            distances <- as.matrix(distances)
            rownames(distances) <- colnames(distances)
            distances_sparse <- as(distances, "dgCMatrix")
            nn <- as.Graph(x = distances_sparse)
            slot(nn, name = "assay.used") <- "RNA"
            nn
        }, error = function(e) {  # We already increase the size limit
            message("Error while transfering NN Graph: ", e$message)
            return(NULL) })

        if (!is.null(nn)) {
            output.obj@graphs$RNA_nn <- nn
            output.obj@graphs$RNA_nn@assay.used <- "RNA"
        }


    }

    saveRDS(output.obj, opt$out)
} else {
    stop("Only read and write operations are permitted")
}
