# Installation

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
