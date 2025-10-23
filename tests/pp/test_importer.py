import dotools_py as do
import os
import shutil
import anndata as ad


def test_importer():
    os.makedirs("./tmp/", exist_ok=True)
    path = "./tmp/"
    do.dt.example_10x(path)

    files = ["./tmp/healthy/outs/filtered_feature_bc_matrix.h5"]

    adata = do.pp.importer_py(
        files,
        ids=["Batch1"],
        metadata={"condition": ["healthy"]},
        doublet_tool="Scrublet",
        min_counts=500,
        high_quantile=95,
        min_genes=10,
        max_genes=2000,
    )
    files = os.listdir("./tmp/healthy/outs")
    assert "Vln_PreQC_Batch1.svg" in files
    assert "Vln_PostQC_Batch1.svg" in files
    assert isinstance(adata, ad.AnnData)
    shutil.rmtree("./tmp")
    return


def test_cellbender():
    import os
    os.makedirs("./tmp")
    do.dt.example_10x(path="./tmp/")
    do.pp.run_cellbender(
        cellranger_path="./tmp/",
        # Contains subfolders for every sample map with CellRanger
        output_path="./tmp/",  # Save the output files from CellBender
        samplenames=["healthy"],  # Name of subfolders, if not specified detected automatically
        cuda=False,  # Run on GPU !!Recommended (Can take up to 1 hour)
        cpu_threads=20,  # If not GPU available, control how many CPUs to use
        epochs=150,  # Default is enough
        lr=0.00001,  # Learning Rate
        log=False,  # Generates a log file for each sample with the stdout
    )
