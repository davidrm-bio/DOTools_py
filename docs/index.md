# DoTools

DoTools is a convenient and user-friendly package to streamline common workflows in single-cell RNA sequencing data
analysis using the scverse ecosystem. It provides high-level wrappers and visualisation functions to help efficiently
preprocess, analyze, and interpret single-cell data.

![Overview](https://github.com/davidrm-bio/DOTools_py/blob/0d6c4cf0cd7ce06f9511bde85b92f73befd575a8/docs/_static/figures/DoTools_Overview.png?raw=1)

```{eval-rst}
.. card:: Installation :octicon:`plug;1em;`
    :link: installation
    :link-type: doc

    Check out the installation guide.
```

```{eval-rst}
.. card:: API reference :octicon:`book;1em;`
    :link: api/index
    :link-type: doc

    The API reference contains a detailed description for each function.
```

```{eval-rst}
.. card:: GitHub :octicon:`mark-github;1em;`
    :link: https://github.com/davidrm-bio/DOTools_py

    Found a bug? Checkout our GitHub for the latest implementation.

```

```{toctree}
:caption: 'General'
:hidden: true
:maxdepth: 2

installation
api/index
changelog
references
```

```{toctree}
:caption: 'Use cases'
:hidden: true
:maxdepth: 2

notebooks/index
```

```{toctree}
:caption: 'About'
:hidden: true
:maxdepth: 1

cite
GitHub <https://github.com/davidrm-bio/DOTools_py>
R version <https://github.com/MarianoRuzJurado/DOtools>

```

## Logging and reproducibility
We offer a logging method to allow to track the commands that have been run and
the output of these commands. To activate it run at the beginning of the session:

```python
import dotools_py as do

do.settings.set_kernel_logger("History.log")

# Run at the end to add Session Information.
do.utility.create_report("History.log")
```

## Citation
> t.b.a
