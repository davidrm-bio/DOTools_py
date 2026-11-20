# Tools: `tl`

The tools module {mod}`dotools_py.tl` contains functions for the downstream
analysis and statistical methods.


## Integration
```{eval-rst}
.. module:: dotools_py.tl
.. currentmodule:: dotools_py

.. autosummary::
    :toctree: generated

    tl.integrate_data
    tl.run_scvi
    tl.run_scanvi
    tl.run_harmony
    tl.run_seurat_integration
```

## Clustering and Cell Annotation
```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.umap_clustering
    tl.auto_annot
    tl.reclustering
    tl.full_recluster
```

## Statistical Analysis
```{eval-rst}
.. autosummary::
    :toctree: generated

    tl.DGEAnalysis
    tl.rank_genes_groups
    tl.rank_genes_condition
    tl.rank_genes_pseudobulk
    tl.rank_genes_consensus
    tl.grouped_ttest
    tl.run_mast
    tl.go_analysis
```


