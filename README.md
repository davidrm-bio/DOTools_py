# DOTools_py

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/davidrm-bio/DOTools_py/test.yaml?branch=main
[badge-docs]:  https://img.shields.io/readthedocs/DOTools_py

Convenient functions for sc/snRNA-seq analysis and visualisation.

## Getting started

Please refer to the [documentation](https://dotools-py.readthedocs.io/en/latest/index.html),
in particular, the [API documentation](https://dotools-py.readthedocs.io/en/latest/api/index.html).

## Installation

You need to have Python 3.10 or newer installed on your system.

There are several alternative options to install DOTools_py:

<-----
1. Install the latest release of `DOTools_py` from [PyPI][]:
    ```bash
    pip install DOTools_py
    ```
---->

1. Install the latest development version:
    ```bash
    pip install git+https://github.com/davidrm-bio/DOTools_py.git@main
    ```

We also have a R implementation of the  [DOTools](https://github.com/MarianoRuzJurado/DOtools). This can be
installed with `devtools`:

```R
devtools::install_github("MarianoRuzJurado/DOtools")
```

## Requirements

Some methods are run through R and require additional dependencies
including: `Seurat`, `MAST`, `scDblFinder`, `zellkonverter` and `optparse`.

```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

install.packages("optparse")
remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)  # Seurat
BiocManager::install("MAST")
BiocManager::install("scDblFinder")
BiocManager::install("zellkonverter")
```

## Release notes

See the [changelog][].

## Contact
Raising up an issue in this Github repository might be the fastest way of submitting suggestions and bugs.
Alternatively you can write to my email: [rodriguezmorales@med.uni-frankfurt.de](mailto:rodriguezmorales@med.uni-frankfurt.de).



## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/davidrm-bio/DOTools_py/issues
[tests]: https://github.com/davidrm-bio/DOTools_py/actions/workflows/test.yaml
[documentation]: https://DOTools_py.readthedocs.io
[changelog]: https://DOTools_py.readthedocs.io/en/latest/changelog.html
[api documentation]: https://DOTools_py.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/DOTools_py
