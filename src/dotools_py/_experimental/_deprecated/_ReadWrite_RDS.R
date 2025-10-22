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


if (opt$operation == 'read') {  # Convert RDS to AnnData
    message("Convert RDS Object to AnnData Object")
    input.obj <- readRDS(opt$input)
    output.obj <- TransformObjectType(input.obj, "SingleCellExperiment")
    writeH5AD(output.obj, opt$out)

    # Transfer missing information
    if (is(input.obj, "Seurat")) {
        tmp_folder <- strsplit(opt$input, "/")[[1]]
        tmp_folder <- tmp_folder[-length(tmp_folder)]
        tmp_folder <- paste(tmp_folder, collapse = "/")

        # Get Variable Features
        hvg <- VariableFeatures(input.obj)
        if (length(hvg) >0) {
            write.csv(as.data.frame(hvg), paste0(tmp_folder, "/VariableFeatures.csv"))
        }

        # Get SNN
        assay_name <- names(input.obj@assays)
        graph_names <- names(input.obj@graphs)
        for (assay in graph_names) {
            if (grepl("_snn", assay)) {
                snn <- as.matrix(input.obj@graphs[[assay]])
                data.table::fwrite(snn, paste0(tmp_folder, "/Connectivities.csv")) # Save SNN
            }
            else if  (grepl("_nn", assay)) {
                nn <- as.matrix(input.obj@graphs[[assay]])
                data.table::fwrite(nn, paste0(tmp_folder, "/Distances.csv"))  # Save NN
            }
        }
    }


} else if (opt$operation == 'write') {  # Convert AnnData to RDS
    message("Convert AnnData Object to RDS Object")
    input.obj <- readH5AD(opt$input)
    output.obj <- TransformObjectType(input.obj, opt$type)

    # Transfer missing information
    if (opt$type == "SeuratObject") {

        # Rename Assay to RNA
        message("Generating RNA assay")
        assay_name <- names(output.obj@assays)
        output.obj[["RNA"]] <- output.obj[[assay_name]]
        DefaultAssay(output.obj) <- "RNA"
        output.obj[[assay_name]] <- NULL

        # Replace orig.ident with batch_key
        message("Saving batch information")
        output.obj$orig.ident <- output.obj@meta.data[opt$batch_key]

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
        }

        # Connectivities -> snn
        message("Getting SNN Graph")
        snn <- tryCatch({
            connectivities <- data.table::fread(paste0(tmp_folder, "/Connectivities.csv"),
                                                check.names = FALSE, stringsAsFactors = FALSE)
            connectivities <- as.matrix(connectivities)

           if ("V1" %in% colnames(connectivities)){
                connectivities$V1 <- NULL
            }
            rownames(connectivities) <- colnames(connectivities)
            connectivities_sparse <- as(connectivities, "dgCMatrix")
            snn <- as.Graph(x = connectivities_sparse)
            slot(snn, name = "assay.used") <- "RNA"

           snn

        }, error = function(e) {
            message("Error while transfering SNN Graph: ", e$message)
            return(NULL) })

        if (!is.null(snn)) {
            output.obj@graphs$RNA_snn <- snn
        }

        # distances -> nn
        message("Getting NN Graph")
        nn <- tryCatch({
            distances <- data.table::fread(paste0(tmp_folder, "/Distances.csv"), check.names = FALSE,
                                           stringsAsFactors = FALSE)
            if ("V1" %in% colnames(distances)){
                distances$V1 <- NULL
            }
            distances <- as.matrix(distances)
            rownames(distances) <- colnames(distances)
            distances_sparse <- as(distances, "dgCMatrix")
            nn <- as.Graph(x = distances_sparse)
            slot(nn, name = "assay.used") <- "RNA"
            nn
        }, error = function(e) {
            message("Error while transfering NN Graph: ", e$message)
            return(NULL) })

        if (!is.null(nn)) {
            output.obj@graphs$RNA_nn <- nn
        }
    }

    saveRDS(output.obj, opt$out)
} else {
    stop("Only read and write operations are permitted")
}
