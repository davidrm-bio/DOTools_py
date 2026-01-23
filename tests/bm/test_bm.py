import matplotlib.pyplot as plt
import pandas as pd

import dotools_py as do


def test_bm():
    adata = do.dt.example_10x_processed()
    adata_unintegrated = adata.copy()

    del adata_unintegrated.obsm["X_CCA"]

    score, ax = do.bm.eval_integration(
        adata,
        adata_unintegrated,
        batch_key="batch",
        annotation_key="annotation",
        use_rep="X_CCA",
        compute_metrics = ["GraphConnectivity", "pcr_comparison", "silhouette_batch", "silhouette_global"],
        show=False

    )
    plt.close()
    assert isinstance(ax, dict)
    assert isinstance(score, pd.DataFrame)
    return
