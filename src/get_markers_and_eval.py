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
from nsforest import nsforesting
from nsforest import evaluating as ev


###### FUNCTION DEFINITIONS START ######
def nsforest_preprocessing(adata: ad.AnnData, data_id: str, cluster_header: str):
    # SCOPE: GLOBAL
    """_summary_

    Args:
        adata (ad.AnnData): _description_
        data_id (str): _description_
        cluster_header (str): _description_

    Returns:
        _type_: _description_
    """
    print(f"Running NS-Forest preproccessing on {data_id}")
    adata = pp.prep_medians(adata=adata, cluster_header=cluster_header, positive_genes_only=True)    
    adata = pp.prep_binary_scores(adata=adata, cluster_header=cluster_header)
    
    return adata

def main():
    
    #### argparse ####
    parser = argparse.ArgumentParser(description="Run NSForest to get global, local and combined markers from data")
    parser.add_argument("--data_id", type=str, required=True, help="String to ID the data")
    parser.add_argument("--path_to_ingested_h5ad", type=str, required=True, help = "Path containing ingested h5ad file produced by ingest.py. Must point to any viable h5ad file.")
    parser.add_argument("--cluster_header", type=str, required=True, help = "Column name of adata.obs that contains cell type labels of interest")
    parser.add_argument("--binary_thresholding", type=str, default = "BinaryFirst_high", help = "Thresholding level for selecting positively expressed genes for Random Forest")
    parser.add_argument("--results_dir", type=str, required=True, help = "Path to save results. Directory named after --data_id.")
    parser.add_argument("--cluster_labels", type=str, required=True, help="Array of endothelial labels")
    parser.add_argument("--n_cores", type=int, required=True, help = "How many cores/CPUs allocated for running NSForest")

    args = parser.parse_args()

    data_id = args.data_id
    h5ad_path = args.path_to_ingested_h5ad
    cluster_header = args.cluster_header
    results_dir = args.results_dir
    endo_labels = [str(cluster_label) for cluster_label in args.cluster_labels.split(",")]
    binary_thresh = args.binary_thresholding
    njobs = args.n_cores
    
    # call functions to run processes

    # READ AND PREPROCESS GLOBAL DATA
    adata = sc.read_h5ad(h5ad_path)
    
    global_adata = adata.copy()
    global_adata = nsforest_preprocessing(adata=global_adata, data_id=data_id, cluster_header=cluster_header)

    # create results/tables/ subdir before first call of NS-Forest
    tables_subdirpath = os.path.join(results_dir, "tables")
    os.makedirs(tables_subdirpath, exist_ok=True)
    
    ###################### MARKER SET DISCOVERY ######################
    
    print("DISCOVERING GLOBAL MARKERS\n")
    global_data_results = nsforesting.NSForest(
        adata = global_adata, cluster_header=cluster_header, output_folder=tables_subdirpath, outputfilename_prefix=f"{cluster_header}_global_NSForest_res",
        gene_selection=binary_thresh, save_supplementary=False, n_jobs=njobs
    )
    
    # PREP MARKERS_DICT FOR GLOBAL MARKERS AND PLOT WHOLE DATA DOTPLOT
    global_markers = {
                    cluster : list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
                    for cluster, markers in zip(global_data_results['clusterName'], global_data_results['NSForest_markers'])
            } #RULE: marker dicts are made right before actually being used; same case with subsets of data
            
    global_markers_endo_only = {
        cluster: global_markers[cluster]
        for cluster in global_markers
        if cluster in endo_labels
        }
    
    print("DISCOVERING CLASS MARKER(S)\n")
    class_adata = adata.copy()
    endo_class_mapping = {"Endothelial" : endo_labels}
    endo_class_mapping = {ct: group for group, types in {**endo_class_mapping}.items() for ct in types}
    class_adata.obs['class_plus_granular'] = class_adata.obs[cluster_header].astype(str).replace(endo_class_mapping).astype('category')
    
    class_data_results = nsforesting.NSForest(adata = class_adata, cluster_header="class_plus_granular", output_folder=tables_subdirpath, 
                                              outputfilename_prefix="class_and_global_NSForest_results",
                                              gene_selection=binary_thresh, save_supplementary=False, n_jobs=njobs)
    
    print("DISCOVERING LOCAL MARKERS\n")
    local_adata = adata[adata.obs[cluster_header].isin(endo_labels)].copy()
    local_adata.obs[cluster_header] = local_adata.obs[cluster_header].cat.remove_unused_categories()
    
    local_adata = nsforest_preprocessing(adata = local_adata, data_id=data_id, cluster_header=cluster_header)
    
    local_data_results = nsforesting.NSForest(
        adata = local_adata, cluster_header=cluster_header, output_folder=tables_subdirpath, outputfilename_prefix=f"{cluster_header}_local_NSForest_res",
        gene_selection=binary_thresh, save_supplementary=False, n_jobs = njobs
        )
    
    ###################### MARKER SET EVALUATION ######################
    # so we have global on global res, local on local, and no combined marker results
    
    print("EVALUATING GLOBAL MARKERS ON LOCAL DATA\n")
    global_marker_on_local_data_res = ev.DecisionTree(adata=local_adata, cluster_header=cluster_header, markers_dict=global_markers_endo_only, combinations = False,
                                                         use_mean=False, save_supplementary=False, output_folder=tables_subdirpath, 
                                                         outputfilename_prefix="global_markers_eval_on_local_data_results")
    
    print("EVALUATING LOCAL MARKERS ON GLOBAL DATA\n")
    local_markers = {
        cluster: list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
        for cluster, markers in zip(local_data_results['clusterName'], local_data_results['NSForest_markers'])
    }
    
    local_marker_on_global_data_res = ev.DecisionTree(adata=global_adata, cluster_header=cluster_header, markers_dict=local_markers, combinations=False, use_mean=False,
                                                         save_supplementary=False, output_folder=tables_subdirpath, 
                                                         outputfilename_prefix="local_markers_eval_on_global_data_results")
    
    print("EVALUATING COMBINED MARKER SETS ON GLOBAL DATA\n")
    class_marker = class_data_results.loc[class_data_results['clusterName'] == 'Endothelial', 'NSForest_markers'].values[0]
    class_marker = list(ast.literal_eval(class_marker)) if isinstance(class_marker, str) else list(class_marker)
    combined_markers = {
        cluster: class_marker + list(markers)
        for cluster, markers in local_markers.items()
    }
    combined_markers_on_global_data = ev.DecisionTree(adata=global_adata, cluster_header=cluster_header, markers_dict=combined_markers, use_mean=False,
                                                         save_supplementary=False, output_folder=tables_subdirpath, 
                                                         outputfilename_prefix="combined_markers_eval_on_global_data_results")
    
    print("EVALUATING COMBINED MARKER SETS ON LOCAL DATA\n")
    combined_markers_on_local_data = ev.DecisionTree(adata=local_adata, cluster_header=cluster_header, markers_dict=combined_markers, use_mean=False,
                                                         save_supplementary=False, output_folder=tables_subdirpath, 
                                                         outputfilename_prefix="combined_markers_eval_on_local_data_results")
###### FUNCTION DEFINITIONS END ######

if __name__ == "__main__":
    main()