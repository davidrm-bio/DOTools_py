import dotools_py as do
import os

def test_gc():
    do.utility.free_memory()
    return None


def test_settings():
    try:
        do.settings.matplotlib_backend("pycharm")
    except Exception:
        do.settings.matplotlib_backend("agg")
    do.settings.matplotlib_backend("agg")
    return

def test_transferLabels():
    adata = do.dt.example_10x_processed()

    adata_subset = do.get.subset(adata, obs_key="annotation", obs_groups="B_cells", copy=True)
    adata_subset.obs["news"] = "testing"
    do.utility.transfer_labels(adata_original=adata,
                               adata_subset=adata_subset,
                               original_key="annotation",
                               subset_key="news",
                               original_labels=["B_cells"])
    assert "testing" in adata.obs["annotation"].unique()

    # Using copy
    adata = do.dt.example_10x_processed()
    adata_subset = do.get.subset(adata, obs_key="annotation", obs_groups="B_cells", copy=True)
    adata_subset.obs["news"] = "testing"
    adata = do.utility.transfer_labels(adata_original=adata,
                               adata_subset=adata_subset,
                               original_key="annotation",
                               subset_key="news",
                               original_labels=["B_cells"],
                               copy=True)
    assert "testing" in adata.obs["annotation"].unique()

    return  None


def test_add_gene_metadata():
    adata = do.dt.example_10x_processed()
    adata = do.utility.add_gene_metadata(data=adata, gene_key="var_names", species="human", add_gene_id=True)
    cols = {'biotype', 'locations', 'gene_id'}
    assert cols.issubset(adata.var.columns)
    df = adata.var.copy()
    df.reset_index(inplace=True)
    df = do.utility.add_gene_metadata(data=df, gene_key="index", species="human", add_gene_id=True)
    cols = {'biotype', 'locations', 'gene_id'}
    assert cols.issubset(adata.var.columns)

    try:
        do.utility.add_gene_metadata(data=["testing"])
    except Exception:
        pass


    return None


def test_spatial():
    adata = do.dt.example_10x_processed()
    sp = False
    if sp:
        do.utility.add_smooth_kernel(adata)
        do.utility.select_slide(adata, s="slide1")


def test_report():
    do.settings.set_kernel_logger(filename="./history.log")
    files = os.listdir("./")
    assert "history.log" in files
    do.settings.toogle_kernel_logger(False)
    # do.utility.create_report("./history.log")
    os.remove("./history.log")
    return



def test_live_display():
    import time

    @do.utility.live_display(current=1, total=1)
    def step():
        time.sleep(2)
    try:
        step()
    except Exception as e:
        print("Failed for live_display", e)
