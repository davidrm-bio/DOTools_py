# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html


## Version 0.0.3 {small}`2026-01-30`

### Features

- Add `dotools_py.pl.heatmap_foldchange` to visualize log2foldchanges between groups across conditions.
- Add ``io`` module for reading/writing. `dotools_py.utility.read_rds` and `dotools_py.utility.save_rds` have been moved to this module.
- Make internal functions in ``dotools_py.pp`` and ``dotools_py.tl`` public.
- Add `dotools_py.get.layer_swap` to swap layers.
- Add `dotools_py.settings.set_kernel_logger` and :func:`dotools_py.settings.toggle_kernel_logger` to record the kernel history.
- Add ``dotools_py.bm`` module that contain metrics for the evaluation of quality control steps.
- Add `dotools_py.pl.density` to visualize density of features in embeddings.
- Add ``random_state`` parameters to the methods.
- Add `dotools_py.utlity.create_report`
- Add `dotools_py.tl.DGEAnalysis`, a class with DGE analysis methods.
- Add `dotools_py.pl.ridgeplot`

### Bug fixes

- Fix Bug in ``dotools_py.pl.barplot``,  ``dotools_py.pl.boxplot`` and  ``dotools_py.pl.violinplot`` where the legends
were not correctly display when `hue` was set but `hue_order` was not set.
- Embedding plots will be saved using a ``vector_friendly`` (scatter plots will use png backend even when exporting as PDF or SVG).
- Internal bug fixes in several other methods
- Harmony runs through the pytorch implementation.

## Version 0.0.2 {small}`2025-11-25`

Correction of bugs and update parameters naming for consistency across functions.

## Version 0.0.1 {small}`2025-10-23`

Pre-release of DoTools, a convenient and user-friendly package to streamline common workflows in single-cell RNA
sequencing data analysis using the scverse ecosystem.
