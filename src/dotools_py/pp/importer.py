import os
import shutil
import uuid

from datetime import date
from pathlib import Path
import subprocess
from typing import Union

import scanpy as sc
import anndata as ad
import numpy as np
import polars
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import logger
from utils import get_paths_utils, convert_path


def _qc_vln(adata: ad.AnnData,
            title: str = 'ViolinPlots - Quality Metrics',
            path: str = None,
            filename: str = 'ViolinPlots.png',
            stats: list = ('total_counts', 'n_genes', 'pct_counts_mt'),
            colors: Union[str, list] = 'lightsteelblue',
            ) -> None:
    """Violin Plots showing basic QC stats

    Generate ViolinPlots to show the distribution of total counts, number of genes and percentage of mitochondrial genes.
    :param adata: anndata object
    :param title: Title of the Plot. Default ViolinPlot
    :param path: path to figure folder. Default ./ (current folder)
    :param filename: name of the file with the plot. Default ViolinPlot.png
    :param stats: obs column name to plot
    :param colors: colors for the violinplots
    :return:
    """
    if isinstance(stats, tuple):
        stats = list(stats)

    assert all(col in list(adata.obs.columns) for col in stats), 'column name in col_obs missing in adata.obs'
    assert len(stats) == 3, 'Expected 3 variables to plot: total_counts, n_genes_by_counts, pct_counts_mt'

    if isinstance(colors, str):
        colors = [colors]
    if len(colors) == 1:
        colors = colors * 3

    fig, axs = plt.subplots(1, 3, figsize=(5, 6))
    for idx in range(3):
        vln = sns.violinplot(adata.obs[stats[idx]], ax=axs[idx], color=colors[idx])
        vln.set_xticklabels([f'Median = {np.round(np.median(adata.obs[stats[idx]]), 1)}'], fontweight='bold')
        vln.set_title('')
    plt.suptitle(title, fontsize=30, fontweight='bold')

    if path is not None:
        plt.savefig(convert_path(path) / filename, bbox_inches='tight')
    return


def _filter_quantiles(adata: ad.AnnData,
                      low: int = None,
                      high: int = None,
                      ) -> ad.AnnData:
    """Filter cells based on total nUMI counts using quantiles
    :param adata: anndata object
    :param low: lower quantile
    :param high: upper qauntile
    :return: anndata object
    """
    counts = adata.obs['total_counts']
    mask = np.ones(adata.n_obs, dtype=bool)
    if low:
        mask &= counts > np.percentile(counts, low)
    if high:
        mask &= counts < np.percentile(counts, high)
    return adata[mask, :].copy()


def _run_scdblfinder(adata: ad.AnnData,
                     batch_key: str = None,
                     ) -> None:
    """Find doublets
    The inference is performed using `scDblFinder <https://github.com/plger/scDblFinder>`_
    in R.
    :param adata: annndata object
    :param batch_key: .obs column name with batch information. Required if the anndata contain more than 1 sample
    :return: None
    """
    logger.info('Finding Neotypic doublets')
    rscript = get_paths_utils('_run_scDblFinder.R')
    tmpdir_path = Path('/tmp') / f"scDblFinder_{uuid.uuid4().hex}"
    tmpdir_path.mkdir(parents=True, exist_ok=False)
    adata.write(tmpdir_path / 'adata_tmp.h5ad')

    logger.info('Running scDblFinder')
    cmd = ['Rscript', rscript, '--input=' + str(tmpdir_path) + '/adata_tmp.h5ad', '--out=' + str(tmpdir_path) + '/']
    if batch_key: cmd = cmd['--name=' + batch_key]
    subprocess.call(cmd)

    doublets = polars.read_csv(tmpdir_path / 'scDblFinder_inference.csv', infer_schema_length=0)
    doublets = doublets.to_pandas()
    doublets = doublets.set_index(adata.obs_names)  # Avoid ImplicitModificationWarning
    adata.obs[['doublet_class', 'doublet_score']] = doublets.values
    shutil.rmtree(tmpdir_path)
    return


def _normalise(adata: ad.AnnData,
               n_reads: int = 10_000,
               max_val: float = None,
               scale: bool = True):
    """Data Normalisation

    The input is an unnormalise anndata object. The data in .X will be log-normalise to 10,000 reads
    per cell. The returned anndata object will contain 3 layers:

    * counts: contains the raw unnormalised counts
    * logcounts: contains the log-normalise counts
    * scaled: contained the log-normalise counts scaled

    Additionally, the log-normalise counts will also be saved under the .X attribute.

    :param adata: annData object
    :param n_reads: target number of reads per cell to normalize to. (Default  is **10,000**)
    :param max_val: maximum expression value after scaling. (Default is **None**)
    :param scale: whether to scale or not the data. (Default is **True**)
    :return: log-normalise anndata object
    """

    adata.layers['counts'] = adata.X.copy()  # Save raw counts
    sc.pp.normalize_total(adata, target_sum=n_reads)
    sc.pp.log1p(adata)
    adata.layers['logcounts'] = adata.X.copy()

    if scale:
        logger.info('Scaling data')
        sc.pp.scale(adata, zero_center=True, max_value=max_val)
        adata.layers['scaled'] = adata.X.copy()
        adata.X = adata.layers['logcounts'].copy()
    return


def _qc_scrna(adata: ad.AnnData,
              ids: str,
              qc_path: str = None,
              batch_key=None,
              min_genes_in_cell: int = 300,
              min_cells_with_genes: int = 5,
              cut_mt: int = 5,
              min_counts: int = None,
              max_counts: int = None,
              min_genes: int = None,
              max_genes: int = None,
              low_quantile: int = None,
              high_quantile: int = None,
              include_rbs: bool = True,
              remove_doublets: bool = False,
              metrics: bool = True,
              copy: bool = False,
              ) -> ad.AnnData:
    """**Quality Control**

    The input is an unprocessed anndata object. The following filtering steps are applied:

    * Filter genes express in low number of cells
    * Filter cells with low number of genes
    * Filter cells with high mitochondrial content. Recommendation: 5 % for scRNA and 3 % for snRNA

    There are two modes for filtering cells based on UMI and Feature counts:

    * Absolute filtering: set absolute values for the maximum and minimum number of UMI and features
    * Quantile filtering: filter out the top and bottom quantile

    Optionally, you can also remove doublets (recommended).

    A ExcelSheet will be generated by default with stats on how many cells and features were removed in
    each step. Additionally, a violin plot showing the distribution of total_counts, n_genes and mt_content
    per cell before and after the quatily control will be generated.

    :param adata: anndata object
    :param ids: id or name for the data
    :param qc_path: path where to save the metric and the violin plots
    :param min_genes_in_cell: minimum number of genes in a cell
    :param min_cells_with_genes:  minimum number of cells expressing a gene
    :param cut_mt: maximum number of mitochondrial content for cells
    :param min_counts: minimum number of counts per cell
    :param max_counts: maximum number of counts per cell
    :param min_genes: minimum number of genes per cell
    :param max_genes: maxinum number of genes per cell
    :param low_quantile: low quantile to filter genes and counts
    :param high_quantile: upper quantile to filter genes and counts
    :param is_mouse: whether input is mouse or not
    :param include_rbs: calculate stats for ribosomal genes
    :param filter_absolute: filter by absolute values
    :param filter_quantile: filter by quantile
    :param remove_doublets: remove doublets
    :param metrics: whether to generate a metrics file or not
    :param copy: make a copy to not modify the input
    :return: processed anndata
    """

    if copy:
        adata = adata.copy()  # Changes not in place

    # Create a metrics file
    today = date.today().strftime("%y%m%d")
    metrics_filename = f'{today}_Metrics_{ids}.xlsx'
    df = pd.DataFrame([], columns=['QC_Step', 'nCells', 'nFeatures', 'Comments'])
    df.loc[0] = ['Input_Shape', adata.shape[0], adata.shape[1], '']

    # Compute Metrics
    mt_gene, ribo_gene = 'mt-', ('rbs', 'rpl')
    qc_metrics = ['mt', 'ribo'] if include_rbs else ['mt']
    adata.var['genenames'] = adata.var_names.str.lower()  # Generalise for any gene format
    adata.var['mt'] = adata.var['genenames'].str.startswith(mt_gene)  # Annotate mitochondria genes
    adata.var['ribo'] = adata.var['genenames'].str.startswith(ribo_gene)  # Annotate mitochondria genes
    sc.pp.calculate_qc_metrics(adata, qc_vars=qc_metrics, percent_top=None, log1p=True, inplace=True, parallel=True)

    # Vln Plots showing Metrics before qc
    _qc_vln(adata, title=f'PreQC for {ids}', path=qc_path, filename=f'Vln_PreQC_{ids}.svg')

    # Step 1 -
    logger.info('Remove Cells with low number of genes')
    sc.pp.filter_cells(adata, min_genes=min_genes_in_cell, inplace=True)
    df.loc[1] = ['Rm_poor_Cells', adata.shape[0], adata.shape[1], 'Remove cells with low number of genes']

    # Step 2 -
    logger.info('Remove Genes lowly expressed')
    sc.pp.filter_genes(adata, min_cells=min_cells_with_genes, inplace=True)
    df.loc[2] = ['Rm_low_Genes', adata.shape[0], adata.shape[1],
                 f'Remove genes express in less than {min_cells_with_genes} cells']

    # Step 3 -
    logger.info('Remove cells with high MT-content')
    adata = adata[adata.obs.pct_counts_mt < cut_mt, :].copy()
    df.loc[3] = ['Rm_cell_high_MT', adata.shape[0], adata.shape[1],
                 f'Remove cells with >{cut_mt}% of Mitochondrial genes']

    # Step 4 -
    logger.info('Remove cells based on nUMI counts')
    assert (min_counts is None) != (low_quantile is None), 'Set min_count or low_quantile'
    assert (max_counts is None) != (high_quantile is None), 'Set max_count or high_quantile'

    if min_counts is not None:
        sc.pp.filter_cells(adata, min_counts=min_counts)
    if max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=max_counts)
    if min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=min_genes)
    if max_genes is not None:
        sc.pp.filter_cells(adata, max_genes=max_genes)

    # Apply quantile-based filtering (conditionally)
    adata = _filter_quantiles(adata, low_quantile, high_quantile)
    df.loc[4] = ['Rm_Cells_nFeatures', adata.shape[0], adata.shape[1],
                 'Remove cells based on nUMI counts and nFeatures']

    # Step 5 -
    if remove_doublets:
        adata.layers['counts'] = adata.X.copy()  # needed for scDblFinder
        _run_scdblfinder(adata, batch_key)
        n_doublets = adata.obs['doublet_class'].value_counts()['doublet']
        adata = adata[adata.obs['doublet_class'] == 'singlet'].copy()
        logger.info(f'Remove {n_doublets} doublets')
        df.loc[5] = ['Rm_doublets', adata.shape[0], adata.shape[1], 'Remove neotypic doublets']

    # Save Metrics File
    if metrics is True:
        df_plot = df.iloc[:, :-1].melt(id_vars='QC_Step')  # Exclude comments
        fig, axs = plt.subplots(1, 1, figsize=(5, 6))  # initializes figure and plots
        bp = sns.barplot(df_plot, hue='QC_Step', x='value', y='variable',
                         order=['nCells', 'nFeatures'],
                         hue_order=list(df['QC_Step']),
                         palette='tab20', ax=axs)

        for container in bp.containers:
            bp.bar_label(container)
        bp.set_title('')
        bp.set_ylabel('', fontsize=18)
        bp.set_xlabel('Counts', fontsize=18)
        bp.legend(title='QC_Step', fontsize=12, frameon=False, title_fontproperties={'weight': 'bold', 'size': 15})
        plt.savefig(os.path.join(qc_path, f'{today}_QC_Metrics{ids}.svg'), bbox_inches='tight')

        # Save Metric File
        df.to_excel(os.path.join(qc_path, metrics_filename), index=False)
    return adata


def importer_py(paths: list,
                ids: list,
                metadata: dict = None,
                batch_key: str = 'batch',
                remove_doublets: bool = True,
                min_genes_in_cell: int = 300,
                min_cells_with_genes: int = 5,
                cut_mt: int = 5,
                n_reads: int = 10_000,
                min_counts: int = None,
                max_counts: int = None,
                min_genes: int = None,
                max_genes: int = None,
                low_quantile: int = None,
                high_quantile: int = None,
                ) -> dict:
    """**sc/snRNA Quality Control**

    The input is a list with the paths to the .h5 files generated with CellRanger, CellBender or StarSolo and
    a list with the ids (batch name) for each sample. You can provide a dictionary with extra metadata to be
    added to the AnnData in the following format:

    ```{python}

    paths = ['/path/sample1,h5', '/path/sample2,h5']

    ids = ['sample1', 'sample2']

    metadata = {'condition': ['wt', 'disease'],
                'age': ['20m', '20m']}
    ```

    The order should always be kept. For each sample a quality control will be applied that includes:

    * Filter genes express in low number of cells
    * Filter cells with low number of genes
    * Filter cells with high mitochondrial content. Recommendation: 5 % for scRNA and 3 % for snRNA
    * Filter cells based on UMI and Features. There are two modes:
        * Absolute filtering: set absolute values for the maximum and minimum number of UMI and features
        * Quantile filtering: filter out the top and bottom quantile
    * Remove doublets using scDblFinder

    A ExcelSheet will be generated by default with stats on how many cells and features were removed in
    each step. Additionally, a violin plot showing the distribution of total_counts, n_genes and mt_content
    per cell before and after the quatily control will be generated. This files will be saved under the folder
    containing the .h5 files.

    After QC, the data will be normalised and scaled (optional) and the highly variable genes will be
    calculated.

    :param paths: list of paths of the .h5 files
    :param ids: list of ids (batch name) for each .h5 file
    :param metadata: dictionary with metadata information
    :param batch_key: column name in .obs for the batch information
    :param remove_doublets: remove doublets with scDblFinder
    :param min_genes_in_cell: minimum number of genes per cell
    :param min_cells_with_genes: minimum cells expressing a genes
    :param n_reads: target sum after normalisation per cell
    :param cut_mt: maximum percentage of mitochondrial genes per cell
    :param min_counts:  minimum number of counts per cell
    :param max_counts: maximum number of counts per cell
    :param min_genes: minimum number of genes per cell
    :param max_genes: maximum number of genes per cell
    :param low_quantile: low quantile to filter cells based on counts and genes
    :param high_quantile: upper quantile to filter cells based on counts and genes
    :return: anndata object with samples concatenated
    """

    # Checks
    assert isinstance(paths, list) and isinstance(ids, list), 'Please provide a list of paths and ids'
    assert len(paths) == len(ids), f'Provided {len(paths)} paths and {len(ids)} ids'

    adata_dict = {}
    for idx, path in enumerate(paths):
        # Save QC Plots in the folder with raw data
        qc_path = convert_path('/'.join(path.split("/")[:-1]))

        logger.info(f'Reading {ids[idx]}')
        try:
            adata = sc.read_10x_h5(path)  # Works for 10x and CellBender and StarSolo?
        except IsADirectoryError:
            adata = sc.read_10x_mtx(path)  # Directory with .mtx and .tsv files

        adata.var_names_make_unique()

        # Add ID and Metadata
        adata.obs[batch_key] = ids[idx]
        if metadata:
            for key, value in metadata.items():
                adata.obs[key] = adata.obs[batch_key].map(dict(zip(ids, value)))

        # Quality Control
        adata = _qc_scrna(
            adata=adata,
            ids=ids[idx],
            batch_key=batch_key,
            qc_path=qc_path,
            metrics=True,
            copy=True,
            min_genes_in_cell=min_genes_in_cell,
            min_cells_with_genes=min_cells_with_genes,
            cut_mt=cut_mt,
            min_counts=min_counts,
            max_counts=max_counts,
            min_genes=min_genes,
            max_genes=max_genes,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            include_rbs=True,
            remove_doublets=remove_doublets
        )

        # Vln Plots showing Metrics before qc
        _qc_vln(adata, title=f'PostQC for {ids[idx]}',
                path=qc_path, filename=f'Vln_PostQC_{ids[idx]}.svg')

        adata_dict[ids[idx]] = adata

    logger.info('Concatenating samples')
    adata_concat = ad.concat(adata_dict.values(), label=batch_key, keys=adata_dict.keys(),
                             join='outer', index_unique='-', fill_value=0)
    logger.info('Normalisation of the expression')
    _normalise(adata_concat, n_reads=n_reads, scale=True)

    logger.info('Finding Highly Variable Genes shared across samples')
    sc.pp.highly_variable_genes(adata_concat, batch_key=batch_key)

    logger.info('Run PCA')
    sc.pp.pca(adata_concat)
    return adata_concat
