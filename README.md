# DOTools_py
<img src="https://github.com/davidrm-bio/DOTools_py/blob/8b7e9988713cc4867443b22c1688b4ae85966ae3/docs/figures/LogoDoTools.png?raw=1" align="right" width="210" class="no-scaled-link"/>

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![Coverage][badge-coverage]][codecoverage]
[![Issues][badge-issues]][issue tracker]
[![Stars][badge-stars]](https://github.com/davidrm-bio/DOTools_py/stargazers)
[![PyPI][badge-pypi]][pypi]
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/dotools-py?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/dotools-py)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/davidrm-bio/DOTools_py/test.yaml?branch=main
[badge-docs]:  https://img.shields.io/readthedocs/DOTools_py/stable
[badge-issues]: https://img.shields.io/github/issues/davidrm-bio/DOTools_py
[badge-stars]: https://img.shields.io/github/stars/davidrm-bio/DOTools_py?style=flat&logo=github&color=yellow
[badge-coverage]: https://codecov.io/gh/davidrm-bio/DOTools_py/branch/main/graph/badge.svg
[badge-pypi]: https://img.shields.io/pypi/v/DOTools_py.svg


Convenient and user-friendly package to streamline common workflows in single-cell RNA sequencing data analysis using
the scverse ecosystem. It provides high-level wrappers and visualisation functions to help efficiently preprocess,
analyze, and interpret single-cell data.

## Getting started

Please refer to the [documentation](https://dotools-py.readthedocs.io/en/stable/index.html),
in particular, the [API documentation](https://dotools-py.readthedocs.io/en/stable/api/index.html).

## Installation

You need to have Python 3.10 or newer installed on your system. We recommend creating
a dedicated [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment.

```bash
conda create -n do_py11 python=3.11
conda activate do_py11
```

There are several alternative options to install DOTools_py:

1. Install the latest release of `DOTools_py` from [PyPI](https://pypi.org/project/DOTools-py/):
```bash
pip install dotools-py
```

2. Install the latest development version:
```bash
pip install git+https://github.com/davidrm-bio/DOTools_py.git@main
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
python -m ipykernel install --user --name=do_py11 --display-name=do_py11
```

## Requirements

Some methods are run through R and require additional dependencies
including: `Seurat`, `MAST`, `scDblFinder`, `zellkonverter`, `data.table` and `optparse`.

```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

install.packages("optparse", Ncpus=8)
install.packages('remotes', Ncpus=8)
install.packages('data.table', Ncpus = 8)
remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)  # Seurat
BiocManager::install("MAST")
BiocManager::install("scDblFinder")
BiocManager::install("zellkonverter")
BiocManager::install('glmGamPoi')
```

For old CPU architectures there can be problems with [polars](https://docs.pola.rs/) making the kernel die
when importing the package. In this case run

```bash
pip install --no-cache polars-lts-cpu
```

We also have an R implementation of the  [DOTools](https://github.com/MarianoRuzJurado/DOtools). This can be
installed with `devtools`:

```R
devtools::install_github("MarianoRuzJurado/DOtools")
```

## Release notes

See the [changelog][].

## Contact
Raising up an issue in this GitHub repository might be the fastest way of submitting suggestions and bugs.
Alternatively you can write to my email: [rodriguezmorales@med.uni-frankfurt.de](mailto:rodriguezmorales@med.uni-frankfurt.de).

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/davidrm-bio/DOTools_py/issues
[tests]: https://github.com/davidrm-bio/DOTools_py/actions/workflows/test.yaml
[documentation]: https://dotools-py.readthedocs.io/en/stable/
[changelog]: https://dotools-py.readthedocs.io/en/latest/changelog.html
[api documentation]: https://dotools-py.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/DOTools-py/
[codecoverage]: https://codecov.io/gh/davidrm-bio/DOTools_py
[down]: https://pepy.tech/project/dotools-py
