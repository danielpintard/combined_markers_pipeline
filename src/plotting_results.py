import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import gc
import copy

from nsforest import preprocessing as pp

###### FUNCTION DEFINITIONS START ######
def compute_and_get_dendrogram_order(adata: ad.AnnData, cluster_header: str, save_path: str, filename_suffix: str):
    # SCOPE: GLOBAL
    """    
    """
    os.makedirs(os.path.join(save_path, "figures", "dendrograms"), exist_ok=True)
    ad = adata.copy()
    pp.dendrogram(ad, cluster_header, save=True, output_folder=save_path, outputfilename_suffix=f"{filename_suffix}_{cluster_header}")
    dendrogram_obj = copy.deepcopy(ad.uns[f"dendrogram_{cluster_header}"])
    del ad
    gc.collect()
    
    return dendrogram_obj

def barplot_nsf_res(df:pd.DataFrame, 
                    value_vars:list, 
                    figsize:tuple = (8,4), 
                    save_path:str = None, 
                    save:bool = True):
    # SCOPE: GLOBAL
    """
    """
    
    
    melted_df = df.melt(id_vars = 'clusterName', value_vars=value_vars, var_name = 'classification_metric', value_name='value')
    
    plt.figure(figsize=figsize, 
            #    constrained_layout = True
               )
    metrics_barplot = sns.barplot(data = melted_df, x = 'clusterName', y = 'value', hue='classification_metric', legend = 'full')
    metrics_barplot.set_xticklabels(melted_df['clusterName'].unique(), rotation = 35, ha = 'right')
    metrics_barplot.legend(loc = 'center left', bbox_to_anchor=(1.00, 0.5))
    
    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi = 150, bbox_inches='tight')
    else:
        plt.show()
        
def metric_comparison_barplots():
    pass

def plot_dotplots_for_global_res(dotplot_dirpath, global_adata, local_adata, global_marker_dict, cluster_label_marker_dict, cluster_header, cluster_labels, marker_type_str, data_type_str):
    # SCOPE: THIS PROJ
    """
    """
    sc.settings.figdir = dotplot_dirpath
    
    # global dotplot
    sc.pl.dotplot(
        global_adata,
        var_names=global_marker_dict,
        groupby=cluster_header,
        dendrogram=True,
        standard_scale='var',
        use_raw=False,
        save=f"{marker_type_str}_markers_on_global_data_full.png"
    )
    
    # abbreviated/truncated dotplot
    sc.pl.dotplot(
        global_adata,
        var_names= cluster_label_marker_dict,
        groupby="subtypes_plus_others", # NOTE: hard-coded
        dendrogram=True,
        standard_scale='var',
        use_raw=False,
        save=f"{marker_type_str}_markers_on_{data_type_str}_data_abbreviated.png"
    )
    
    # local data dotplot
    sc.pl.dotplot(
        local_adata, 
        var_names=cluster_label_marker_dict,
        groupby= cluster_header,
        dendrogram=True,
        standard_scale='var',
        use_raw=False,
        save=f"global_markers_on_local_data.png"
    )
    
def main():
    
    fig_dir = os.path.join(results_dir, 'figures')
    dotplot_dir = os.path.join(fig_dir, 'dotplots')
    dendrogram_dir = os.path.join(fig_dir, 'dendrograms')
    barplot_dir = os.path.join(fig_dir, 'barplots')
    for figure_subdir in [dotplot_dir, dendrogram_dir, barplot_dir]:
        os.makedirs(figure_subdir, exist_ok=True)

###### FUNCTION DEFINITIONS END ######