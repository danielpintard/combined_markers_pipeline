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
                    figsize:tuple = (9,6), 
                    save_path:str = None,
                    plot_title: str = None):
    # SCOPE: GLOBAL
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        value_vars (list): _description_
        figsize (tuple, optional): _description_. Defaults to (8,5).
        save_path (str, optional): _description_. Defaults to None.
        plot_title (str, optional): _description_. Defaults to None.
    """
    melted_df = df.melt(id_vars = 'clusterName', value_vars=value_vars, var_name = 'classification_metric', value_name='value')
    
    plt.figure(figsize=figsize, 
            #    constrained_layout = True
               )
    metrics_barplot = sns.barplot(data = melted_df, x = 'clusterName', y = 'value', hue='classification_metric', legend = 'full',)
    if plot_title:
        metrics_barplot.set_title(plot_title)
    metrics_barplot.set_xticklabels(melted_df['clusterName'].unique(), rotation = 35, ha = 'right')
    metrics_barplot.legend(loc = 'center left', bbox_to_anchor=(1.00, 0.5))
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi = 150, bbox_inches='tight')
    else:
        plt.show()
        
def metric_comparison_barplots(df1: pd.DataFrame, df1_hue_label: str, df2: pd.DataFrame,
                               df2_hue_label: str, hue_label_field: str,
                               group_name_field: str, metrics_2_plot: list, save_path: str):
    """_summary_

    Args:
        df1 (pd.DataFrame): _description_
        df1_hue_label (str): _description_
        df2 (pd.DataFrame): _description_
        df2_hue_label (str): _description_
        hue_label_field (str): _description_
        group_name_field (str): _description_
        metrics_2_plot (list): _description_

    Returns:
        _type_: _description_
    """
    
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
            hue=hue_label_field,
            palette='viridis',
            order=df.sort_values([metric, hue_label_field], ascending=[False, True])[group_name_field],
            ax=ax
        )

        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
        ax.set_title(f"{hue_label_field} {metric} comparison")

        # keep a single legend outside the last plot; drop the rest
        if i == n_metrics - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1.00, 0.5))
        else:
            ax.legend_.remove()
    
    # is this modification gonna cause issues
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi = 150, bbox_inches = "tight")
    else:
        fig.tight_layout()
        return fig, axes
    
def plot_dotplots_for_results(dotplot_dirpath: str, global_adata: ad.AnnData, marker_dict: dict, 
                              cluster_label_marker_dict: dict, cluster_header: str, 
                              marker_type_str: str, data_type_str: str, local_adata: ad.AnnData = None):
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
    if local_adata is not None: 
        sc.pl.dotplot(
            local_adata, 
            var_names=cluster_label_marker_dict,
            groupby= cluster_header,
            dendrogram=True,
            standard_scale='var',
            use_raw=False,
            save=f"{marker_type_str}_markers_on_local_data.png"
        )
        
def results_to_long_df(df: pd.DataFrame, cluster_labels: list,
                       metrics: list):
    """Reshape wide NS-Forest results into the long format plot_facet_per_dataset expects."""
    d = df.copy()
    d = d[d["clusterName"].isin(cluster_labels)]
    long_df = d.melt(
        id_vars=["clusterName", "markerSet_dataContext"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="score",
    ).rename(columns={"markerSet_dataContext": "condition_label"})
    
    return long_df


def master_comparison_plots(long_df, out_dir, metrics, condition_label_order, col_wrap=3, palette="colorblind"):
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    
    x_order = [c for c in condition_label_order
                if c in set(long_df["condition_label"])]

    # one figure per metric
    for metric in [m for m in metrics if m in set(long_df["metric"])]:
        metric_subset = long_df[long_df["metric"] == metric]

        g = sns.catplot(
            data=metric_subset, kind="bar",
            x="condition_label", y="score",
            hue="condition_label",          # color bars by condition
            order=x_order, hue_order=x_order,
            col="clusterName", col_wrap=col_wrap,
            palette=palette, height=4, aspect=1.4,
            errorbar=None, legend=False,     # x-axis already labels them
            sharex=True, sharey=True,
        )

        g.set(ylim=(0, 1))
        g.set_axis_labels("", metric)
        g.set_titles("{col_name}")

        for ax in g.axes.flat:
            ax.tick_params(axis="x", rotation=45)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

        g.figure.suptitle(f"{metric} by marker set vs eval context",
                            y=1.02)

        fpath = os.path.join(out_dir, f"{metric}_comparison_masterplot.png")
        g.savefig(fpath, dpi=200, bbox_inches="tight")
        plt.close(g.figure)
        saved.append(fpath)

    return saved

def main():
    parser = argparse.ArgumentParser(description="Plot results from produced files in get_markers_and_eval.py")
    parser.add_argument("--data_id", type=str, required=True, help="String to ID the data")
    parser.add_argument("--path_to_ingested_h5ad", type=str, required=True, help = "Path containing ingested h5ad file produced by ingest.py. Must point to any viable h5ad file.")
    parser.add_argument("--cluster_header", type=str, required=True, help = "Column name of adata.obs that contains cell type labels of interest")
    parser.add_argument("--results_dir", type=str, required=True, help = "Path to save results. Directory named after --data_id.")
    parser.add_argument("--cluster_labels", type=str, required=True, help="Array of labels for clusters in lineage/class of interest")
    parser.add_argument("--master_results_filepath", type=str, required=True, help="Path to csv containing all of the produced NS-Forest results from get_markers_and_eval.py")
    
    args = parser.parse_args()
    
    data_id = args.data_id
    h5ad_path = args.path_to_ingested_h5ad
    cluster_header = args.cluster_header
    results_dir = args.results_dir
    cluster_labels = [str(cluster_label) for cluster_label in args.cluster_labels.split(",")]
    
    fig_dir = os.path.join(results_dir, 'figures')
    dotplot_dir = os.path.join(fig_dir, 'dotplots')
    dendrogram_dir = os.path.join(fig_dir, 'dendrograms')
    barplot_dir = os.path.join(fig_dir, 'barplots')
    reports_dir = os.path.join(results_dir, 'tables', 'reports')
    for subdir in [dotplot_dir, dendrogram_dir, barplot_dir, reports_dir]:
        os.makedirs(subdir, exist_ok=True)
        
    
    # READ IN H5AD, CREATE ANNOTATIONS THAT WILL BE HANDY FOR PLOTTING AND COMPUTE HIEARCHICAL CLUSTERING
    adata = sc.read_h5ad(h5ad_path) 
    
    local_adata = adata[adata.obs[cluster_header].isin(cluster_labels)].copy()
    local_adata.obs[cluster_header] = local_adata.obs[cluster_header].cat.remove_unused_categories()
    
    # create annotations
    adata.obs['subtypes_plus_others'] = pd.Categorical(np.where(adata.obs[cluster_header].isin(cluster_labels), adata.obs[cluster_header], 'Other Cell Types'))
    endo_class_mapping = {"Endothelial" : cluster_labels}
    endo_class_mapping = {ct: group for group, types in {**endo_class_mapping}.items() for ct in types}
    adata.obs['class_plus_granular'] = adata.obs[cluster_header].astype(str).replace(endo_class_mapping).astype('category')
    
    # compute hiearchical clustering and plot dendrograms
    adata.uns[f"dendrogram_{cluster_header}"] = compute_and_get_dendrogram_order(
        adata=adata, cluster_header=cluster_header, save_path=results_dir, filename_suffix="global_data")
    local_adata.uns[f"dendrogram_{cluster_header}"] = compute_and_get_dendrogram_order(
        adata=local_adata, cluster_header=cluster_header, save_path=results_dir, filename_suffix="local_data")
    adata.uns[f"dendrogram_class_plus_granular"] = compute_and_get_dendrogram_order(
        adata=adata, cluster_header="class_plus_granular", save_path=results_dir, filename_suffix="global_data_w_endothelial_class")
    
    # READ IN NS-FOREST RESULTS DATA TO CONSTRUCT MARKER DICTIONARIES FOR DOTPLOTS
    all_results_df = pd.read_csv(args.master_results_filepath)
    markerSet_dataContexts = all_results_df['markerSet_dataContext'].unique().tolist()
    metrics = ['f_score', 'precision', 'recall', 'onTarget']
    condition_label_order = [
        "global_markers_on_global_data",
        "global_markers_on_local_data",
        "local_markers_on_local_data",
        "local_markers_on_global_data",
        "combined_markers_on_local_data",
        "combined_markers_on_global_data"
    ]
    
    for results_set in markerSet_dataContexts:
        res_df = all_results_df[all_results_df['markerSet_dataContext'] == results_set].copy()
        res_df['markers'] = res_df['markers'].apply(ast.literal_eval)
        
        if results_set == "class_markers_on_global_data": # conditional logic for handling the dotplots for class markers
            class_marker = res_df.loc[res_df['clusterName'] == 'Endothelial', 'markers'].values[0]
            class_marker = list(ast.literal_eval(class_marker)) if isinstance(class_marker, str) else list(class_marker)
            res_dict = {
                cluster : list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
                for cluster, markers in zip(res_df['clusterName'], res_df['markers'])
                }
            _res_dict = {}
            for cluster, marker in res_dict.items():
                if cluster == "Endothelial":
                    for label in cluster_labels:
                        _res_dict[label] = marker
                else:
                    _res_dict[cluster] = marker
            
            sc.settings.figdir = os.path.join(results_dir, 'figures', 'dotplots')
            sc.pl.dotplot(
                adata, 
                var_names = _res_dict, 
                groupby = cluster_header,
                dendrogram = True,
                standard_scale = 'var',
                use_raw = False,
                save = f'{results_set}.png'
            )
            
            sc.pl.dotplot(
                adata, 
                var_names = _res_dict, 
                groupby = "class_plus_granular",
                dendrogram = True,
                standard_scale = 'var',
                use_raw = False,
                save = f'{results_set}_class_plus_granular.png'
            )
        else: # conditional logic for other dotplots and their respective NS-Forest metrics barplots
            marker_type = results_set.split("markers_on_")[0]
            data_type = results_set.split("_markers_on_")[1].split("_")[0]
            res_dict = dict(zip(res_df['clusterName'], res_df['markers']))
            clusters_only_dict = {
                cluster: markers
                for cluster, markers in res_dict.items() if cluster in cluster_labels
            }
            # create dotplots
            plot_dotplots_for_results(dotplot_dirpath = dotplot_dir, global_adata = adata, marker_dict = res_dict, cluster_label_marker_dict = clusters_only_dict,
                cluster_header = cluster_header, marker_type_str = marker_type, data_type_str = data_type, local_adata = local_adata)
            
            # create barplots
            cl_lbls_only_res_df = res_df[res_df['clusterName'].isin(cluster_labels)]
            barplot_nsf_res(df=cl_lbls_only_res_df, value_vars=metrics, save_path=os.path.join(barplot_dir, f"{results_set}_metrics_barplot.png"))
    
    """
    Pseudocode for generating reports text files
    NOTE: These report files aim to answer the following questions:
    1) do global markers perform worse in local context? (global_markers_on_global_data vs global_markers_on_local_data) - different reports for each metric
    2) are local markers performant on global data? (local_markers_on_local_data vs local_markers_on_global_data)
    3) are combined markers actually better than global_markers_on_local_data (global_markers_on_local_data vs combined_markers_on_local_data) and local_markers_on_global_data (local_markers_on_global_data vs combined_markers_on_global_data)?
    """
    markerSet_metric_comparisons = [
        ("global_markers_on_global_data", "global_markers_on_local_data"), # do global markers perform worse in local context?
        ("local_markers_on_local_data", "local_markers_on_global_data"), # are local markers performant on global data?
        ("global_markers_on_local_data", "combined_markers_on_local_data"), # are combined markers actually better than global_markers_on_local_data, i.e. provide happy medium
        ("local_markers_on_global_data", "combined_markers_on_global_data") # are combined markers actually better than local_markers_on_global_data, i.e. provide happy medium   
    ]
    
    for comparison in markerSet_metric_comparisons:
        df1 = all_results_df[all_results_df['markerSet_dataContext'] == comparison[0]]
        df1 = df1[df1['clusterName'].isin(cluster_labels)]
        df2 = all_results_df[all_results_df['markerSet_dataContext'] == comparison[1]]
        df2 = df2[df2['clusterName'].isin(cluster_labels)]
        metric_comparison_barplots(df1 = df1, df1_hue_label = comparison[0], df2 = df2, df2_hue_label = comparison[1],
                                       hue_label_field='markerSet_dataContext', group_name_field='clusterName',
                                       metrics_2_plot=metrics, save_path=os.path.join(barplot_dir, f"{comparison}_metrics_barplot.png"))
    
    long_df = results_to_long_df(df=all_results_df, cluster_labels=cluster_labels, metrics=metrics)
    master_comparison_plots(long_df=long_df, out_dir=barplot_dir, metrics=metrics, condition_label_order=condition_label_order)  
    
###### FUNCTION DEFINITIONS END ######

if __name__ == "__main__":
    main()