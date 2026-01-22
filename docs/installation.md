# Installation

You need to have Python 3.10 or newer installed on your system. We recommend creating
a dedicated [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) environment.

```bash
conda create -n scrna_py11 python=3.11 -y
conda activate scrna_py11
```

There are several alternative options to install DOTools_py:

1. Install the latest release of `DOTools_py` from [PyPI](https://pypi.org/project/DOTools-py/):
```bash
pip install uv
uv pip install dotools-py
```

2. Install the latest development version:
```bash
pip install git+https://github.com/davidrm-bio/DOTools_py.git@main
```

Finally, to use this environment in jupyter notebook, add jupyter kernel for this environment:

```bash
python -m ipykernel install --user --name=scrna_py11 --display-name=scrna_py11
```

## Requirements

This package has been tested on macOS, Linux and Windows System. For a standard dataset (e.g., 6 samples with 10k cells each)
we suggest 16GB of RAM and at least 5 CPUs.

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

# R version

We also have an R implementation of the  [DOTools](https://github.com/MarianoRuzJurado/DOtools). This can be
installed from Bioconductor:

```R
if (!requireNamespace("BiocManager", quietly=TRUE)) {
    install.packages("BiocManager")
}
BiocManager::install("DOtools")
devtools::install_github("MarianoRuzJurado/DOtools")
```

The developmental version can be downloaded using `devtools`:

```R
devtools::install_github("MarianoRuzJurado/DOtools", ref="devel")
```
