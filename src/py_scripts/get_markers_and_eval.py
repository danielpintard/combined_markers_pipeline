# NOTE: perhaps considering option to add custom dendrogram order??
# NOTE: For testing purposes, I should have this script report at what datetime were data objects deleted to clear memory, because that would be great for crossreferencing
#       with the memory usage overtime for a given job, ex. if I see a drop in memory usage, that could suggest an adata object being deleted is the reason why
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import argparse
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import gc
import copy
warnings.filterwarnings('ignore')

import nsforest as ns
from nsforest import preprocessing as pp
from nsforest import nsforesting
from nsforest import plotting as pl
from nsforest import evaluating as ev

#### argparse ####
parser = argparse.ArgumentParser(description="Run NSForest to get global, local and combined markers from data")
parser.add_argument("--data_id", type=str, required=True, help="String to ID the data")
parser.add_argument("--tmpdir", type=str, required=True, help = "Temporary space for holding intermediate files. On Biowulf, set $TMPDIR to lscratch space.")
parser.add_argument("--cluster_header", type=str, required=True, help = "Column name of adata.obs that contains cell type labels of interest")
parser.add_argument("--binary_thresholding", type=str, default = "BinaryFirst_high", help = "Thresholding level for selecting positively expressed genes for Random Forest")
parser.add_argument("--results_dir", type=str, required=True, help = "Path to save results. Directory named after --data_id.")
parser.add_argument("--endo_labels", type=str, nargs='+', required=True, help="Array of endothelial labels")
parser.add_argument("--n_cores", type=int, required=True, help = "How many cores/CPUs allocated for running NSForest")

args = parser.parse_args()

data_id = args.data_id
tmpdir = args.tmpdir
cluster_header = args.cluster_header
results_dir = args.results_dir
endo_labels = args.endo_labels
binary_thresh = args.binary_thresholding
njobs = args.n_cores

###### FUNCTION DEFINITIONS ######

def barplot_nsf_res(df:pd.DataFrame, 
                    value_vars:list, 
                    figsize:tuple = (8,4), 
                    save_path:str = None, 
                    save:bool = True):
    
    melted_df = df.melt(id_vars = 'clusterName', value_vars=value_vars, var_name = 'classification_metric', value_name='value')
    
    plt.figure(figsize=figsize, constrained_layout = True)
    metrics_barplot = sns.barplot(data = melted_df, x = 'clusterName', y = 'value', hue='classification_metric', legend = 'full')
    metrics_barplot.set_xticklabels(melted_df['clusterName'].unique(), rotation = 35, ha = 'right')
    metrics_barplot.legend(loc = 'center left', bbox_to_anchor=(1.00, 0.5))
    
    if save:
        plt.tight_layout()
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
    else:
        plt.show()

# read adata from tmpdir, need to get list of objects from tmp dir and then deploy them as batch array
adata = sc.read_h5ad(os.path.join(tmpdir, f'{data_id}_tmp_files', 'h5ads', f'{data_id}_ingested.h5ad'))

##################################################################################
################################ RUNNING NSFOREST ################################
##################################################################################

###################### GET GLOBAL MARKERS FOR GLOBAL DATASET ######################
print("###################### GETTING GLOBAL MARKERS FOR GLOBAL DATASET ######################\n")
global_adata = adata.copy()

os.makedirs(os.path.join(results_dir, 'figures', 'dendrograms'), exist_ok=True)
#### create and save dendrogram of global data. 'X_pca' should be precomputed in a controlled manner
ns.pp.dendrogram(
    global_adata,
    cluster_header=cluster_header,
    save=True,
    output_folder=os.path.join(results_dir, 'figures', 'dendrograms'),
    outputfilename_suffix=f"global_{cluster_header}"
)

#### NSForest preprocessing - prep cluster median exp and binary scores for genes per cluster
global_adata = ns.pp.prep_medians(global_adata, cluster_header=cluster_header, positive_genes_only=True)
global_adata = ns.pp.prep_binary_scores(global_adata, cluster_header=cluster_header)

### Running NSForest
## MAKEDIRS: ENSURE `tables` dir is ready to recieve NSForest results
os.makedirs(os.path.join(results_dir, 'tables/'), exist_ok=True)
global_dataset_res = nsforesting.NSForest(
    global_adata,
    cluster_header=cluster_header,
    output_folder=os.path.join(results_dir, 'tables/'),
    outputfilename_prefix=f"{cluster_header}_global_NSF",
    gene_selection=binary_thresh,
    n_binary_genes=10, # may make a param user can config in master_script.sh
    save_supplementary=False, # need to check how much this 'clogs' output dirs + how helpful this info is,
    # save = True, # comment out when using v4.0
    n_jobs = njobs
)

## MAKEDIRS: ENSURE `dotplots` dir is made
os.makedirs(os.path.join(results_dir, 'figures', 'dotplots'), exist_ok=True)

### Plotting global markers on global dataset
print("Plotting (dotplot) global markers on global dataset...\n")

global_markers = {
    cluster : list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
    for cluster, markers in zip(global_dataset_res['clusterName'], global_dataset_res['NSForest_markers'])
    }

adata.uns[f'dendrogram_{cluster_header}'] = copy.deepcopy(global_adata.uns[f'dendrogram_{cluster_header}'])

del global_adata
gc.collect()

sc.settings.figdir = os.path.join(results_dir, 'figures', 'dotplots')
sc.pl.dotplot(
    adata, 
    var_names = global_markers, 
    groupby = cluster_header,
    dendrogram = True,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_global_markers_global_data.png',
)

### Plotting global markers on local dataset
print("Plotting (dotplot) global markers on local dataset...\n")

endo_markers = {
    cluster : global_markers[cluster]
    for cluster in endo_labels
    if cluster in global_markers
}

#extract dendrogram order for endo_labels in dendrogram
global_categories = adata.obs[cluster_header].cat.categories
global_dendro_indices = adata.uns[f'dendrogram_{cluster_header}']['categories_idx_ordered']
ordered_global_clusters = [global_categories[i] for i in global_dendro_indices]
endo_dendro_order = [cluster for cluster in ordered_global_clusters if cluster in endo_labels]

sc.pl.dotplot(
    adata[adata.obs[cluster_header].isin(endo_labels)], 
    var_names = endo_markers, 
    groupby = cluster_header,
    categories_order=endo_dendro_order,
    dendrogram = False, # sure this will be false because it will err out if not, but I do still want to borrow dendrogram order from global
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_global_markers_local_data.png',
)

### Plotting (barplot) classification metrics of global markers on local data
print("Plotting (barplot) classification metrics of global markers on local data...\n")
endo_global_res = global_dataset_res[global_dataset_res['clusterName'].isin(endo_labels)].sort_values(by='clusterName')
### MAKEDIRS: ENSURE `barplots` dir is created
os.makedirs(os.path.join(results_dir, 'figures', 'barplots'), exist_ok=True)
barplot_nsf_res(
    endo_global_res, 
    # value_vars=["f_score", "precision", "recall"], # `precision` for NS-Forest v4.1
    value_vars=["f_score", "PPV", "recall"], # `PPV` for NS-Forest v4.0
    save=True, 
    save_path=os.path.join(results_dir, 'figures', 'barplots', f"{data_id}_global_marker_metrics.png")
    )

###################### GET LOCAL MARKERS FOR LOCAL DATASET ######################
print("###################### GETTING LOCAL MARKERS FOR LOCAL DATASET ######################\n")
local_adata = adata[adata.obs[cluster_header].isin(endo_labels)].copy()

ns.pp.dendrogram(
    local_adata,
    cluster_header=cluster_header,
    save=True,
    output_folder=os.path.join(results_dir, 'figures', 'dendrograms'),
    outputfilename_suffix=f"local_{cluster_header}"
)

#### NSForest preprocessing - prep cluster median exp and binary scores for genes per cluster
local_adata = ns.pp.prep_medians(local_adata, cluster_header=cluster_header, positive_genes_only=True)
local_adata = ns.pp.prep_binary_scores(local_adata, cluster_header=cluster_header)

local_dataset_res = nsforesting.NSForest(
    local_adata,
    cluster_header=cluster_header,
    output_folder=os.path.join(results_dir, 'tables/'),
    outputfilename_prefix=f"{cluster_header}_local_NSF",
    gene_selection=binary_thresh,
    n_binary_genes=10, # may make a param user can config in master_script.sh
    # save=True, # comment out when using v4.0
    n_jobs = njobs
)

### Plotting local markers on local dataset
print("Plotting (dotplot) local markers on local dataset...\n")

local_markers = {
    cluster : list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
    for cluster, markers in zip(local_dataset_res['clusterName'], local_dataset_res['NSForest_markers'])
    }

sc.settings.figdir = os.path.join(results_dir, 'figures', 'dotplots')
sc.pl.dotplot(
    local_adata, 
    var_names = local_markers, 
    groupby = cluster_header,
    dendrogram = True,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_local_markers_local_data.png')

del local_adata
gc.collect()

# Plotting metrics for local markers on local data
print("Plotting metrics for local markers on global data...\n")
barplot_nsf_res(
    local_dataset_res,
    # value_vars=['f_score', 'precision'],
    value_vars=['f_score', 'PPV'],
    save=True,
    save_path=os.path.join(results_dir, 'figures', 'barplots', f"{data_id}_local_on_local_marker_metrics.png")
)


### Plotting local markers on global dataset -- accompanied evaulation metrics calculated in RUNNING EVALUATION section (barplot is there as well)
print("Plotting (dotplot) local markers on global dataset...\n")

# inserting endo local markers into global results
df_endo_local = local_dataset_res.set_index('clusterName')
df_global = global_dataset_res.set_index('clusterName')
df_global.update(df_endo_local)
global_plus_endo_local_res = df_global.reset_index()

# marker dict with local endo markers and global for eveything else
global_plus_endo_local_markers = {
    cluster : list(ast.literal_eval(markers)) if isinstance(markers, str) else list(markers)
    for cluster, markers in zip(global_plus_endo_local_res['clusterName'], global_plus_endo_local_res['NSForest_markers'])
    }

sc.settings.figdir = os.path.join(results_dir, 'figures', 'dotplots')
sc.pl.dotplot(
    adata, 
    var_names = global_plus_endo_local_markers, 
    groupby = cluster_header,
    dendrogram = True,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_local_markers_global_data.png'
)

###################### GET CLASS MARKERS FOR GLOBAL DATASET ######################
print("###################### GETTING CLASS MARKERS FOR GLOBAL DATASET ######################\n")
class_adata = adata.copy()
endo_class_mapping = {"Endothelial" : endo_labels}
endo_class_mapping = {ct: group for group, types in {**endo_class_mapping}.items() for ct in types}
class_adata.obs['class_plus_granular'] = class_adata.obs[cluster_header].astype(str).replace(endo_class_mapping).astype('category')

#### NSForest preprocessing - prep cluster median exp and binary scores for genes per cluster
class_adata = ns.pp.prep_medians(class_adata, cluster_header='class_plus_granular', positive_genes_only=True)
class_adata = ns.pp.prep_binary_scores(class_adata, cluster_header='class_plus_granular')

class_dataset_res = nsforesting.NSForest(
    class_adata,
    cluster_header='class_plus_granular',
    output_folder=os.path.join(results_dir, 'tables/'),
    outputfilename_prefix=f"class_and_global_NSF",
    gene_selection=binary_thresh,
    n_binary_genes=10, # may make a param user can config in master_script.sh
    save_supplementary=False, # need to check how much this 'clogs' output dirs + how helpful this info is,
    # save = True, # comment out when using v4.0
    n_jobs = njobs
)

del class_adata
gc.collect()

# Plotting (dotplot) class marker on global dataset
print("Plotting (dotplot) class marker on global dataset...\n")

endo_class_marker = class_dataset_res.loc[class_dataset_res['clusterName'] == 'Endothelial', 'NSForest_markers'].values[0]
class_markers = list(ast.literal_eval(endo_class_marker)) if isinstance(endo_class_marker, str) else list(endo_class_marker)
global_plus_class_markers = {
    cluster : class_markers if cluster in endo_labels else markers
    for cluster, markers in global_markers.items()
}

sc.settings.figdir = os.path.join(results_dir, 'figures', 'dotplots')
sc.pl.dotplot(
    adata, 
    var_names = global_plus_class_markers, 
    groupby = cluster_header,
    dendrogram = True,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_class_markers_global_data.png'
)

##################################################################################
################################ RUNNING EVALUATION ##############################
##################################################################################

###################### EVALUATE LOCAL MARKERS ON GLOBAL DATASET ######################
print("###################### EVALUATE LOCAL MARKERS ON GLOBAL DATASET ######################\n")
eval_adata_1 = adata.copy()

local_on_global_eval_res = ns.ev.DecisionTree(eval_adata_1, 
                                        cluster_header, 
                                        global_plus_endo_local_markers, 
                                        combinations = False, 
                                        use_mean = False,
                                        # save = True, # comment out when using v4.0
                                        save_supplementary = False,
                                        output_folder = os.path.join(results_dir, 'tables/'), 
                                        outputfilename_prefix = "local_marker_eval_on_global")

del eval_adata_1
gc.collect()

# Plotting metrics for local markers on global data -- accompanied dotplot already created and saved
print("Plotting metrics for local markers on global data...\n")
barplot_nsf_res(
    local_on_global_eval_res[local_on_global_eval_res['clusterName'].isin(endo_labels)].sort_values(by='clusterName'),
    # value_vars=['f_score', 'precision'],
    value_vars=['f_score', 'PPV'],
    save=True,
    save_path=os.path.join(results_dir, 'figures', 'barplots', f"{data_id}_local_on_global_marker_metrics.png")
)

###################### EVALUATE COMBINED MARKERS ON GLOBAL DATASET ####################
print("###################### EVALUATE COMBINED MARKERS ON GLOBAL DATASET ####################\n")

combined_endo_markers = {
    cluster : list(class_markers) + list(l_markers)
    for cluster, l_markers in local_markers.items()
}

for cluster, combined_list in combined_endo_markers.items():
    if cluster in df_global.index:
        df_global.at[cluster, 'NSForest_markers'] = str(combined_list)
        
combined_markers_df_global = df_global.reset_index()

combined_markers = {
    cluster : [str(g) for g in ast.literal_eval(markers)] if isinstance(markers, str) else [str(g) for g in markers]
    for cluster, markers in zip(combined_markers_df_global['clusterName'], combined_markers_df_global['NSForest_markers'])
}

eval_adata_2 = adata.copy()

combined_markers_eval_res = ns.ev.DecisionTree(
    eval_adata_2,
    cluster_header,
    combined_markers,
    combinations=False,
    use_mean=False,
    # save=True, # comment out when using v4.0
    save_supplementary=False,
    output_folder = os.path.join(results_dir, 'tables/'), 
    outputfilename_prefix = "combined_markers_eval"
    )

del eval_adata_2
gc.collect()

### Plotting (dotplot) combined markers on global dataset
print("Plotting (dotplot) combined markers on global dataset...\n")
sc.pl.dotplot(
    adata, 
    var_names = combined_markers, 
    groupby = cluster_header,
    dendrogram = True,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_combined_markers_global_data.png'
)

endo_combined_markers = {
    cluster : markers
    for cluster, markers in combined_markers.items() if cluster in endo_labels
}

### Plotting (dotplot) combined markers on local dataset
print("Plotting (dotplot) combined markers on local dataset...\n")
sc.pl.dotplot(
    adata[adata.obs[cluster_header].isin(endo_labels)], 
    var_names = endo_combined_markers, 
    categories_order=endo_dendro_order,
    groupby = cluster_header,
    dendrogram = False,
    standard_scale = 'var',
    use_raw = False,
    save = f'{cluster_header}_combined_markers_local_data.png'
)

### Plotting (barplot) combined markers classification metrics on local dataset
print("Plotting (barplot) combined markers classification metrics on local dataset...\n")

local_combined_marker_res = combined_markers_eval_res[combined_markers_eval_res['clusterName'].isin(endo_labels)]

barplot_nsf_res(
    local_combined_marker_res,
    # value_vars=['f_score', 'precision'],
    value_vars=['f_score', 'PPV'],
    save=True,
    save_path=os.path.join(results_dir, 'figures', 'barplots', f"{data_id}_combined_markers_metrics.png")
)


#### COMPARISON BARPLOTS
print("Plotting comparison barplots")

global_df = endo_global_res.copy()
loc_on_glob_df = local_on_global_eval_res[local_on_global_eval_res['clusterName'].isin(endo_labels)].copy()
combined_df = combined_markers_eval_res[combined_markers_eval_res['clusterName'].isin(endo_labels)].copy()


global_df['Marker_Strategy'] = 'Global Markers'
combined_df['Marker_Strategy'] = 'Combined Markers'
loc_on_glob_df['Marker_Strategy'] = 'Local on Global'

combined_vs_global = pd.concat([combined_df, global_df], ignore_index=True)
combined_vs_loc_on_glob = pd.concat([combined_df, loc_on_glob_df], ignore_index=True)

### only concerned with plotting combined vs local_on_global performance
# should probably modularize barplot function so that data prep is one function
# so I can have a generalized barplotting function for a case like this

plt.figure(figsize=(6,4))
combined_vs_locglob_p = sns.barplot(
    data=combined_vs_loc_on_glob, 
    x='clusterName', 
    y='f_score', # You can swap this to 'precision' or 'recall' to see those metrics!
    hue='Marker_Strategy',
    palette='viridis' 
)
combined_vs_locglob_p.set_xticklabels(
    combined_vs_loc_on_glob['clusterName'].unique(), 
    rotation = 35,
    ha = 'right'
    )
combined_vs_locglob_p.legend(
    loc = 'center left', 
    bbox_to_anchor=(1.00, 0.5)
    )

plt.savefig(os.path.join(results_dir, 'figures', 'barplots', "combined_vs_loc_on_glob_fscore.png"), dpi=200)

plt.figure(figsize=(6,4))
combined_vs_locglob_p = sns.barplot(
    data=combined_vs_loc_on_glob, 
    x='clusterName', 
    # y='precision',
    y='PPV',
    hue='Marker_Strategy',
    palette='viridis' 
)
combined_vs_locglob_p.set_xticklabels(
    combined_vs_loc_on_glob['clusterName'].unique(), 
    rotation = 35,
    ha = 'right'
    )
combined_vs_locglob_p.legend(
    loc = 'center left', 
    bbox_to_anchor=(1.00, 0.5)
    )

plt.savefig(os.path.join(results_dir, 'figures', 'barplots', "combined_vs_loc_on_glob_precision.png"), dpi=200)