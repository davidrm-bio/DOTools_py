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
                metavar = "character")
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

TransformObjectType <- function (obj, type){
    if (is(obj, "Seurat")){
        obj_type <- "SeuratObject"
    } else if (is(obj, "SingleCellExperiment")) {
        obj_type <- "SingleCellExperiment"
    }

    if (obj_type == type){
        return(obj)  # The class we want is the same
    } else if (obj_type == "SingleCellExperiment" &&  type == "SeuratObject") {
        seu.obj <- as.Seurat(obj, counts = "counts", data = "logcounts")  # We have SCE and want SeuratObject
        return(seu.obj)
    } else if (obj_type == "SeuratObject" &&  type == "SingleCellExperiment") {
        sce <- Seurat::as.SingleCellExperiment(obj)  #We have SeuratObject and want SingleCellExperiment
        return(sce)
    }
}


if (opt$operation == 'read'){  # Convert RDS to AnnData
    message("Convert RDS Object to AnnData Object")
    input.obj <- readRDS(opt$input)
    output.obj <- TransformObjectType(input.obj, "SingleCellExperiment")
    writeH5AD(output.obj, opt$out)

} else if (opt$operation == 'write'){  # Convert AnnData to RDS
    message("Convert AnnData Object to RDS Object")
    input.obj <- readH5AD(opt$input)
    output.obj <- TransformObjectType(input.obj, opt$type)
    saveRDS(output.obj, opt$out)
} else {
    stop("Only read and write operations are permitted")
}
