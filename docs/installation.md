# Installation

You need to have Python 3.10 or newer installed on your system. We recommend creating
a dedicated [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment.

```bash
conda create -n scrna_py11 python=3.11 -y
conda activate scrna_py11
pip install uv
```

There are several alternative options to install DOTools_py:

1. Install the latest release of `DOTools_py` from [PyPI](https://pypi.org/project/DOTools-py/):
```bash
uv pip install dotools-py
```

2. Install the latest development version:
```bash
uv pip install git+https://github.com/davidrm-bio/DOTools_py.git@main
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
python -m ipykernel install --user --name=scrna_py11 --display-name=scrna_py11
```

## Requirements

This package has been tested on macOS, Linux and Windows System. For a standard dataset (e.g., 6 samples with 10k cells each)
we suggest 16GB of RAM and at least 5 CPUs.

Some methods are run through R and require additional dependencies
including: `Seurat`, `MAST`, `scDblFinder`, `anndataR`, `data.table` and `optparse`.

```R
if (!requireNamespace("pak", quietly = TRUE)) {
  install.packages("pak")
}

pak::pkg_install(c(
  "optparse", "remotes",  "data.table", "bioc::MAST", "bioc::scDblFinder",
  "bioc::anndataR",  "bioc::glmGamPoi",  "github::satijalab/seurat@seurat5"
))

```

For old CPU architectures there can be problems with [polars](https://docs.pola.rs/) making the kernel die
when importing the package. In this case run

```bash
uv pip install --no-cache polars-lts-cpu
```

# R version

We also have an R implementation of the  [DOTools](https://github.com/MarianoRuzJurado/DOtools). This can be
installed from Bioconductor:

```R
if (!requireNamespace("pak", quietly=TRUE)) {
    install.packages("pak")
}
pak::pkg_install("bioc::DOtools")
```

The developmental version can be downloaded using `devtools`:

```R
pak::pkg_install("github::MarianoRuzJurado/DOtools@devel")
```
