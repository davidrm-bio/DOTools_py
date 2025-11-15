# Plotting: `pl`

The plotting module {mod}`dotools_py.pl` contains a collection of
functions for enhance visualization of sc/snRNA-seq data.

## sc/snRNA
```{eval-rst}
.. module:: dotools_py.pl
.. currentmodule:: dotools_py

.. autosummary::
    :toctree: generated

    pl.embedding
    pl.split_embeddding
    pl.umap
    pl.dotplot
    pl.heatmap
    pl.barplot
    pl.violinplot
    pl.boxplot
    pl.lineplot
    pl.cell_composition
    pl.split_bar_gsea
    pl.correlation
    pl.volcano_plot
```

## Visium
These functions allow the visualisation for AnnData containing spatial transcriptomics (Visium)
data.
```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.slides
    pl.layers
```

## Classes
These classes allow to calculate and add statistical information to bar-,
box- and violin-plots.
```{eval-rst}
.. autosummary::
    :toctree: generated

    pl.TestData
    pl.StatsPlotter
```
