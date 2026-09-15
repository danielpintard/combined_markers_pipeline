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
    """\
        Computes hiearchical clustering results based on PCA space. Will plot a dendrogram \
        of hiearchical clustering results and save it to path specified by `save_path`.

    Args:
        adata (ad.AnnData): _description_
        cluster_header (str): _description_
        save_path (str): _description_
        filename_suffix (str): _description_

    Returns:
        _type_: _description_
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
        
def metric_comparison_barplots(df1: pd.DataFrame, df1_hue_label: str, df2: pd.DataFrame, df2_hue_label: str, hue_label_field: str,
                               group_name_field: str, metrics_2_plot: list, hue: str):
    
    df1 = df1.copy()
    df1[hue_label_field] = df1_hue_label
    df2 = df2.copy()
    df2[hue_label_field] = df2_hue_label
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    n_metrics = len(metrics_2_plot)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(6 * n_metrics, 4),
        squeeze=False
    )
    axes = axes[0] 

    for i, metric in enumerate(metrics_2_plot):
        ax = axes[i]

        sns.barplot(
            data=df,
            x=group_name_field,
            y=metric,
            hue=hue,
            palette='viridis',
            order=df.sort_values([metric, hue], ascending=[False, True])[group_name_field],
            ax=ax
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
        ax.set_title(metric)

        # keep a single legend outside the last plot; drop the rest
        if i == n_metrics - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.00, 0.5))
        else:
            ax.legend_.remove()

    fig.tight_layout()
    return fig, axes
    
def plot_dotplots_for_results(dotplot_dirpath: str, global_adata: ad.AnnData, local_adata: ad.AnnData, marker_dict: dict, cluster_label_marker_dict: dict, cluster_header: str, marker_type_str: str, data_type_str: str):
    # SCOPE: THIS PROJ
    """\
        This function takes in adata objects for both local and global data, any viable marker_dict \
        and produces dotplots displaying the expression of genes included in marker_dict.

    Args:
        dotplot_dirpath (str): Path to directory where resultant dotplots are saved to.
        global_adata (ad.AnnData): Whole single cell data, containing all cell types.
        local_adata (ad.AnnData): Single cell data only containing lineage/class of interest
        marker_dict (dict): Master marker dictionary
        cluster_label_marker_dict (dict): _description_
        cluster_header (str): _description_
        marker_type_str (str): _description_
        data_type_str (str): _description_
    """
    sc.settings.figdir = dotplot_dirpath
    
    # global dotplot
    sc.pl.dotplot(
        global_adata,
        var_names=marker_dict,
        groupby=cluster_header,
        dendrogram=True,
        standard_scale='var',
        use_raw=False,
        save=f"{marker_type_str}_markers_on_{data_type_str}_data_full.png"
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
        save=f"{marker_type_str}_markers_on_local_data.png"
    )

    
def main():
    
    parser = argparse.ArgumentParser(description="Plot results from produced files in get_markers_and_eval.py")
    parser.add_argument("--data_id", type=str, required=True, help="String to ID the data")
    parser.add_argument("--path_to_ingested_h5ad", type=str, required=True, help = "Path containing ingested h5ad file produced by ingest.py. Must point to any viable h5ad file.")
    parser.add_argument("--cluster_header", type=str, required=True, help = "Column name of adata.obs that contains cell type labels of interest")
    parser.add_argument("--results_dir", type=str, required=True, help = "Path to save results. Directory named after --data_id.")
    parser.add_argument("--cluster_labels", type=str, required=True, help="Array of endothelial labels")
    
    args = parser.parse_args()
    
    data_id = args.data_id
    h5ad_path = args.path_to_ingested_h5ad
    cluster_header = args.cluster_header
    results_dir = args.results_dir
    endo_labels = [str(cluster_label) for cluster_label in args.cluster_labels.split(",")]
    
    fig_dir = os.path.join(results_dir, 'figures')
    dotplot_dir = os.path.join(fig_dir, 'dotplots')
    dendrogram_dir = os.path.join(fig_dir, 'dendrograms')
    barplot_dir = os.path.join(fig_dir, 'barplots')
    for figure_subdir in [dotplot_dir, dendrogram_dir, barplot_dir]:
        os.makedirs(figure_subdir, exist_ok=True)
        
    # READ IN H5AD, CREATE ANNOTATIONS THAT WILL BE HANDY FOR PLOTTING AND COMPUTE HIEARCHICAL CLUSTERING
    adata = sc.read_h5ad(results_dir) 
    
    local_adata = adata[adata.obs[cluster_header].isin(endo_labels)].copy()
    local_adata.obs[cluster_header] = local_adata.obs[cluster_header].cat.remove_unused_categories()
    
    # create annotations
    adata.obs['subtype_plus_others'] = pd.Categorical(np.where(adata.obs[cluster_header].isin(endo_labels), adata.obs[cluster_header], 'Other Cell Types'))
    endo_class_mapping = {"Endothelial" : endo_labels}
    endo_class_mapping = {ct: group for group, types in {**endo_class_mapping}.items() for ct in types}
    adata.obs['class_plus_granular'] = adata.obs[cluster_header].astype(str).replace(endo_class_mapping).astype('category')
    
    # compute hiearchical clustering and plot dendrograms
    adata.uns[f"dendrogram_{cluster_header}"] = compute_and_get_dendrogram_order(
        adata=adata, cluster_header=cluster_header, save_path=results_dir, filename_suffix="global_data")
    local_adata.uns[f"dendrogram_{cluster_header}"] = compute_and_get_dendrogram_order(
        adata=local_adata, cluster_header=cluster_header, save_path=results_dir, filename_suffix="local_data")
    adata.uns[f"dendrogram_class_plus_granular"] = compute_and_get_dendrogram_order(
        adata=adata, cluster_header="class_plus_granular", save_path=results_dir, filename_suffix="global_data_w_endothelial_class")
    
    
    
###### FUNCTION DEFINITIONS END ######

if __name__ == "__main__":
    main()