import shutil
from typing import Literal
from pathlib import Path

import anndata as ad

from dotools_py import logger
from dotools_py.utils import convert_path

class RDSConverter:
    """Class to convert between AnnData and Seurat/SCE.

    This class allows convertion between AnnData, SeuratObject and SingleCellExperiment Object.

    Parameters
    ----------
    input_obj
        The input AnnData Object or the path to an H5AD or RDS Object.
    out_obj
        Format of the output object.
    batch_key
        Column in the AnnData with batch information.
    rds_batch_key
        Column in the Seurat object with batch information.
    path
        The path to the folder where the object will be saved.
    filename
        Name of the output RDS/H5AD file.
    get_anndata
        Whether to return the AnnData or not when using to_h5ad

    Example
    -------

    >>> import os
    >>> import dotools_py as do
    >>> os.makedirs("/tmp/Converter", exist_ok=True)
    >>> adata = do.dt.example_10x_processed()
    >>> converter = do.utility.RDSConverter(input_obj=adata, out_obj="seurat", path="/tmp/Converter", filename="adata.rds")
    >>> converter.to_rds()
    >>> os.listdir("/tmp/Converter")
    ['adata.rds']
    >>> converter = do.utility.RDSConverter(input_obj="/tmp/Converter/adata.rds", out_obj="anndata")
    >>> adata = converter.to_h5ad()
    >>> adata
    AnnData object with n_obs × n_vars = 700 × 1851
    obs: 'nCount_originalexp', 'nFeature_originalexp', 'batch', 'condition', 'n_genes_by_counts',
         'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'total_counts_mt', 'log1p_total_counts_mt',
         'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', 'n_counts', 'doublet_class', 'doublet_score', 'leiden', 'cell_type', 'autoAnnot', 'celltypist_conf_score', 'annotation', 'annotation_recluster', 'ident'
    var: 'highly_variable'
    uns: 'X_name'
    obsm: 'X_cca', 'X_pca', 'X_umap'
    layers: 'logcounts', 'counts'
    obsp: 'connectivities', 'distances'

    """

    def __init__(self,
                 input_obj: str | Path | ad.AnnData = None,
                 out_obj: Literal["anndata", "seurat", "sce"] = "anndata",
                 batch_key: str = "batch",
                 rds_batch_key: str = "orig.ident",
                 path: str | Path = None,
                 filename: str = None,
                 get_anndata: bool = True,
                 ):
        import uuid

        self.input = input_obj
        self.out = out_obj
        self.batch_key = batch_key
        self.path = convert_path(path) if path is not None else None
        self.filename = filename
        self.orig = rds_batch_key
        self.get_anndata = get_anndata

        self._infer_input()

        if self.input_type == "anndata" and not isinstance(self.input, ad.AnnData):
            self.input = ad.read_h5ad(self.input)

        self._get_hvg_graphs()

        tmp_folder = Path("/tmp") / f"Convertion_{uuid.uuid4().hex}"
        tmp_folder.mkdir(parents=True, exist_ok=False)
        self.tmp_folder = tmp_folder
        return

    def __enter__(self):
        from rpy2 import robjects as ro
        ro.r("invisible(gc())")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from rpy2 import robjects as ro
        import shutil
        import gc

        # Clean temporary folder if it still exists
        if hasattr(self, "tmp_folder") and self.tmp_folder.exists():
            shutil.rmtree(self.tmp_folder, ignore_errors=True)

        ro.r("rm(list = ls())")
        ro.r("invisible(gc())")
        gc.collect()


        return False

    def _infer_input(self):
        """Infer the type of the input object.

        :return: Returns None. The input_type and rds_obj attribute will be set.
        """
        from rpy2 import robjects as ro
        rds_obj = None
        if isinstance(self.input, ad.AnnData):
            input_type = "anndata"
        elif isinstance(self.input, str) or isinstance(self.input, Path):
            if ".h5ad" in self.input.lower():
                input_type = "anndata"
            elif ".rds" in self.input.lower():
                rds_obj = ro.r["readRDS"](self.input)
                rds_class = str(ro.r['class'](rds_obj)[0])
                input_type = "seurat" if "seurat" == rds_class.lower() else "sce"
            else:
                raise Exception("Provide a valid Path to *.h5ad or *.rds file")
        else:
            raise Exception("Provide a valid input: AnnData Object or Path")

        self.input_type = input_type
        self.rds_obj = rds_obj
        return

    def _get_hvg_graphs(self):
        import scipy.sparse as sp
        from rpy2 import robjects as ro
        from rpy2.robjects.packages import importr
        SEURAT_OBJ = importr("SeuratObject")

        slot = ro.r["slot"]
        as_matrix = ro.r["as.matrix"]
        names = ro.r["names"]

        hvg, snn, nn = None, None, None
        if self.input_type == "anndata":
            if "distances" in self.input.obsp:
                nn = self.input.obsp["distances"]
            if "connectivities" in self.input.obsp:
                snn = self.input.obsp["connectivities"]
            if "highly_variable" in self.input.var.columns:
                hvg = self.input.var.highly_variable
                hvg = list(hvg[hvg == True].index)
        elif self.input_type == "seurat":
            rds_obj = self.rds_obj
            assays = list(SEURAT_OBJ.Assays(rds_obj))  # Get all Assays
            hvg, snn, nn = [], [], []
            for assay in assays:
                rds_obj = ro.r["DefaultAssay<-"](**{"object": rds_obj, "value": assay})
                if len(hvg) == 0:  # Extract HVGs
                    hvg = list(SEURAT_OBJ.VariableFeatures(rds_obj))
                neighbors_assays = names(slot(rds_obj, "graphs"))
                if len(neighbors_assays) != 0:
                    if assay + "_snn" in neighbors_assays:
                        snn = sp.csr_matrix(as_matrix(slot(rds_obj, "graphs").rx2(assay + "_snn")))
                    if assay + "_nn" in neighbors_assays:
                        nn = sp.csr_matrix(as_matrix(slot(rds_obj, "graphs").rx2(assay + "_nn")))
        else:
            logger.warn("SingleCellExperiment does not contain HighlyVariableFeatures or graphs")

        self.hvg = hvg
        self.snn = snn
        self.nn = nn
        return

    def to_h5ad(self) -> ad.AnnData | None:
        from rpy2 import robjects as ro
        from rpy2.robjects.packages import importr
        ZELLKONVERTER = importr("zellkonverter")

        logger.info("Generating AnnData from RDS")

        if self.input_type == "seurat":  # Convert Seurat to SCE for ZELLKONVERTER
            sce_obj = ro.r["DefaultAssay<-"](**{"object": self.rds_obj, "value": "RNA"})
            as_sce = ro.r["as.SingleCellExperiment"]
            sce_obj = as_sce(sce_obj)
        else:
            sce_obj = self.rds_obj

        # TODO - Change everything to be self-contained
        ZELLKONVERTER.writeH5AD(sce_obj, str(self.tmp_folder / "adata.h5ad"))
        adata = ad.read_h5ad(self.tmp_folder / "adata.h5ad")
        shutil.rmtree(str(self.tmp_folder))

        # Add missing information
        logger.info("Transferring HVGs and Graphs")
        if self.hvg is not None:
            adata.var["highly_variable"] = [True if g in self.hvg else False for g in adata.var_names]
        if self.snn is not None:
            adata.obsp["connectivities"] = self.snn
        if self.nn is not None:
            adata.obsp["distances"] = self.nn

        # Rename reductions
        for key in [key for key in adata.obsm.keys()]:
            new_key = "X_" + key.lower().replace(".", "_").replace("-", "_")
            adata.obsm[new_key] = adata.obsm[key].values
            del adata.obsm[key]

        # Rename orig.ident if present
        if self.orig in list(adata.obs.columns):
            logger.info(f"Renaming orig.ident to {self.batch_key}")
            adata.obs[self.batch_key] = adata.obs[self.orig].copy()
            del adata.obs[self.orig]

        # Default is X with raw counts
        if all(adata.X.data % 1 == 0):
            adata.layers["counts"] = adata.X.copy()

        if self.path is not None:
            assert self.filename is not None, "Provide a filename"
            adata.write(self.path / self.filename)

        if self.get_anndata:
            return adata
        else:
            return None

    def to_rds(self):
        from rpy2 import robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import StrVector
        from rpy2.robjects.packages import importr
        SEURAT_OBJ = importr("SeuratObject")
        ZELLKONVERTER = importr("zellkonverter")
        SEURAT = importr("Seurat")

        assert self.path is not None, "Convertion to RDS without providing path and filename for the output"

        as_seurat = ro.r["as.Seurat"]
        slot = ro.r["slot"]
        names = ro.r["names"]
        get_item = ro.r["[["]

        set_col = ro.r["$<-"]

        if self.rds_obj is None:
            # We come from AnnData
            self.input.write(self.tmp_folder / "adata.h5ad")
            self.input = None  # Clean Memory
            sce_obj = ZELLKONVERTER.readH5AD(str(self.tmp_folder / "adata.h5ad"))
            shutil.rmtree(str(self.tmp_folder))

            if self.out == "seurat":
                logger.info(f"Generating Seurat from AnnData")

                # Rename Assay to RNA
                seu_obj = SEURAT_OBJ.as_Seurat(sce_obj)
                del sce_obj

                old_assay_name = list(SEURAT_OBJ.Assays(seu_obj))[0]  # Should only contain one assay
                if old_assay_name != "RNA":
                    assay_obj = get_item(seu_obj, old_assay_name)
                    seu_obj = self._set_item(seu_obj, "RNA", assay_obj)
                    seu_obj = ro.r["DefaultAssay<-"](**{"object": seu_obj, "value": "RNA"})
                    seu_obj = self._set_item(seu_obj, old_assay_name, ro.NULL)

                # Replace orig.ident with batch_key
                meta_data = slot(seu_obj, "meta.data")
                batch_column = get_item(meta_data, self.batch_key)
                meta_data = set_col(meta_data, self.orig, batch_column)
                seu_obj = self._set_slot(seu_obj, "meta.data", meta_data)

                # Add HVG
                logger.info("Transferring HVGs")
                if self.hvg is not None:
                    hvg_r = StrVector(self.hvg)
                    rna_assay = self._set_slot(slot(seu_obj, "assays").rx2("RNA"), "var.features", hvg_r)
                    assays = slot(seu_obj, "assays")
                    assays[0] = rna_assay
                    seu_obj = self._set_slot(seu_obj, "assays", assays)

                # Rename reductions to remove X_ and make lowercase
                reductions = slot(seu_obj, "reductions")
                reduction_names = names(reductions)
                reduction_names_py = [str(n) for n in reduction_names]
                clean_names = [ro.r["tolower"](ro.r["sub"]("^X_", "", n))[0] for n in reduction_names_py]
                for old_name, new_name in zip(reduction_names_py, clean_names):
                    reduction_obj = get_item(reductions, old_name)
                    reductions = self._set_item(reductions, new_name, reduction_obj)
                    reductions = self._set_item(reductions, old_name, ro.NULL)
                seu_obj = self._set_slot(seu_obj, "reductions", reductions)

                # Add graphs
                logger.info("Transferring Graphs")
                if self.snn is not None and self.nn is not None:
                    snn_graph = self._csr_to_seurat_graph(self.snn, seu_obj, assay_used="RNA")
                    nn_graph = self._csr_to_seurat_graph(self.nn, seu_obj, assay_used="RNA")
                    graphs = slot(seu_obj, "graphs")
                    graphs = self._set_item(graphs, "RNA_snn", snn_graph)
                    graphs = self._set_item(graphs, "RNA_nn", nn_graph)
                    obj_to_save = self._set_slot(seu_obj, "graphs", graphs)
                else:
                    obj_to_save = seu_obj
            elif self.out == "sce":
                logger.info(f"Generating SCE from AnnData")
                obj_to_save = sce_obj
            else:
                raise Exception("Save to_rds but output format is anndata")
        else:
            if self.input_type == "sce":
                logger.info(f"Generating Seurat from SCE")
                obj_to_save = as_seurat(self.rds_obj)
            else:
                raise Exception("Input Object is already a Seurat Object")

        ro.r["saveRDS"](obj_to_save, str(self.path / self.filename))
        del seu_obj
        del obj_to_save
        return

    @staticmethod
    def _csr_to_seurat_graph(csr, seu_obj, assay_used="RNA"):
        from rpy2 import robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri, IntVector, FloatVector, StrVector
        from rpy2.robjects.packages import importr
        import scipy.sparse as sp


        Matrix = importr("Matrix")  # imports Matrix package

        as_graph = ro.r["as.Graph"]

        if not hasattr(csr, "tocsc"):
            csr = sp.csr_matrix(csr.values)

        coo = csr.tocoo()

        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter + numpy2ri.converter):
            i_r = IntVector(coo.row + 1)
            j_r = IntVector(coo.col + 1)
            x_r = FloatVector(coo.data)
            dims_r = IntVector(csr.shape)

        ro.r.assign("seu_obj", seu_obj)
        ro.r("cell_barcodes <- colnames(seu_obj)")
        cell_barcodes = ro.r("cell_barcodes")

        r_dgc = Matrix.sparseMatrix(
            i=i_r,
            j=j_r,
            x=x_r,
            dims=dims_r,
            giveCsparse=True
        )
        ro.r.assign("r_dgc", r_dgc)
        ro.r.assign("rownames", StrVector(cell_barcodes))
        ro.r.assign("colnames", StrVector(cell_barcodes))
        ro.r("dimnames(r_dgc) <- list(rownames, colnames)")
        r_dgc = ro.r("r_dgc")  # retrieve back with dimnames

        graph = as_graph(x = r_dgc)
        graph = RDSConverter._set_slot(graph, "assay.used", ro.StrVector([assay_used]))
        return graph

    @staticmethod
    def _set_item(obj, key, value):
        import rpy2.robjects as ro
        ro.globalenv["obj"] = obj
        ro.globalenv["key"] = key
        ro.globalenv["value"] = value
        ro.r("obj[[key]] <- value")
        return ro.r("obj")

    @staticmethod
    def _set_slot(obj, slot_name, value):
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        """Safely set an S4 slot in R (e.g., Seurat@meta.data)."""

        # Handle None as R NULL
        if value is None:
            ro.r.assign("obj", obj)
            ro.r.assign("slot_name", slot_name)
            ro.r("slot(obj, slot_name) <- NULL")
            return ro.r("obj")

        # Convert pandas DataFrame or NumPy arrays to R equivalents
        with ro.conversion.localconverter(ro.default_converter + pandas2ri.converter):
            value_r = ro.conversion.py2rpy(value)

        # Assign into R environment and set slot safely via R code
        ro.r.assign("obj", obj)
        ro.r.assign("slot_name", slot_name)
        ro.r.assign("value_r", value_r)
        ro.r("slot(obj, slot_name) <- value_r")

        return ro.r("obj")
