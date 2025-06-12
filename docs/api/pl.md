# Plotting `pl`

The plotting module {mod}`dotools_py.pl` contains a collection of
functions for enhance visualisation of sc/snRNA-seq data building on `scanpy`
visualisation methods.

## sc/snRNA
```{eval-rst}
.. module:: dotools_py.pl
.. currentmodule:: dotools_py

.. autosummary::
    :toctree: generated

    pl.dotplot
    pl.embedding
    pl.split_embeddding
    pl.umap
    pl.cell_props
    pl.split_bar_gsea
    pl.volcano_plot
    pl.barplot
    pl.violin
    pl.boxplot
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
