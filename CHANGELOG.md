# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html


## Version 0.0.3

### Features

- Add ``dotools_py.pl.heatmap_foldchange`` to visualize log2foldchanges between groups across conditions.
- Add ``io`` module for reading/writing. `dotools_py.utility.read_rds` and `dotools_py.utility.save_rds` have been moved to this module.
- Add `dotools_py.pp.find_doublets` to detect doublets.
- Add `dotools_py.get.layer_swap` to swap layers.

### Bug fixes

- Fix Bug in ``dotools_py.pl.barplot``,  ``dotools_py.pl.boxplot`` and  ``dotools_py.pl.violinplot`` where the legends
were not correctly display when `hue` was set but `hue_order` was not set.
- Embedding plots will be saved using a ``vector_friendly`` (scatter plots will use png backend even when exporting as PDF or SVG).

## Version 0.0.2 {small}`2025-11-25`

Correction of bugs and update parameters naming for consistency across functions.

## Version 0.0.1 {small}`2025-10-23`

Pre-release of DoTools, a convenient and user-friendly package to streamline common workflows in single-cell RNA
sequencing data analysis using the scverse ecosystem.
