# Preprocessing `pp`

The preprocessing module {mod}`do.pp` compiles all the basic quality control steps for
analysing sc/snRNA-seq with different implementations building on [Scanpy](https://github.com/scverse/scanpy) and other tools
such as [scDblFinder](https://github.com/plger/scDblFinder),
[DoubletDetection](https://github.com/JonathanShor/DoubletDetection) and
[CellBender](https://github.com/broadinstitute/CellBender).

```{eval-rst}
.. module:: dotools_py.pp
.. currentmodule:: dotools_py

.. autosummary::
    :toctree: generated

    pp.run_cellbender
    pp.importer_py
    pp.sctransform_normalise
```
