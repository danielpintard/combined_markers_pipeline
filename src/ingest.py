"""
As a practise, I am gonna tag new function I write with their SCOPE to indicate the logical bounds of that function.
The categories of SCOPE are: 
1) THIS SCRIPT - this function can only be used in this script file, i.e. it is virtually \
unusable in any other script and would require a considerable rewrite to be useable in another script
2) THIS PROJ - this function's scope of utility is bounded by this project's directory structure, and can be reused in other scripts in this directory
3) GLOBAL - these are functions that can be trivially used in other scripts. These functions can be imported into another script, used and cause no \
    errors with proper usage. Functions that can be reused in other projects via copy+paste, should only require one or few minimal changes to be \
    reused in a codebase
"""


## CONFIG START
import warnings
warnings.filterwarnings('ignore')

import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import nsforest as ns
from nsforest import preprocessing as pp

import argparse
import scipy.sparse as sp
import os
import gc

seed = 42 # NOTE:seed passing should probably happen higher up in the program architecture, maybe included in params config file or something like that
## CONFIG END

###### FUNCTION DEFINITIONS START ######

def resolve_var_names(adata: ad.AnnData, data_id: str, cxg: bool, var_col: str):
    # SCOPE : THIS PROJ
    """\
        Ensures adata.var.index (aliased as adata.var_names) contains gene symbols. This function is \
        particularly useful when working with adata objects sourced from CellxGene. Also has to deal with \
        renaming "C7_ENSG00000112936" to "C7" in `HLCA_Core`'s adata.var['feature_name']

    Args:
        adata (ad.AnnData): Annotated data matrix.
        data_id (str): String used to identify the dataset.
        cxg (bool): Indicate whether the AnnData object was sourced from CellxGene data portal.

    Returns:
        ad.AnnData: AnnData object with HGNC gene symbols as indices in .var_names.
    """
    
    # needed to be added since in CellxGene, HLCA names this gene as "C7" but the downloaded h5ad that Ajith has for this 
    # dataset names this genes "C7_ENSG00000112936"
    if data_id == 'HLCA_Core':
        adata.var['feature_name'] = adata.var['feature_name'].cat.add_categories(['C7'])
        adata.var.loc[adata.var['feature_name'] == 'C7_ENSG00000112936', 'feature_name'] = 'C7'
        adata.var['feature_name'] = adata.var['feature_name'].cat.remove_unused_categories() 
    
    if cxg:
        adata.var['ensembl_id'] = adata.var_names
        adata.var.index = adata.var['feature_name'].astype(str)
        adata.var_names_make_unique()
        adata.var.index.name = None
    elif var_col == "NONE":
        adata.var_names = adata.var_names
    else:
        var_col = var_col
        adata.var.index = adata.var[var_col].astype(str)
    
    return adata

def validate_cluster_labels(adata: ad.AnnData, cluster_header: str, cluster_labels: list):
    # SCOPE: GLOBAL
    """_summary_

    Args:
        adata (ad.AnnData): _description_
        cluster_header (str): _description_
        cluster_labels (list): _description_

    Raises:
        ValueError: _description_
    """
    present = set(adata.obs[cluster_header].unique())
    missing = [lab for lab in cluster_labels if lab not in present]
    if missing:
        raise ValueError(
            f"Cluster label(s) not found in adata.obs['{cluster_header}']: {missing}. "
            f"Check the labels provided in the sample sheet against this dataset."
        )

def check_X_transformation(adata):
    """\
        Checks adata.X to ensure it has been transformed using scanpy.pp.normalize_total(target_sum=1e4) and scanpy.pp.log1p()

    Args:
        adata (ad.AnnData): Annotated data matrix.

    Returns:
        ad.AnnData: AnnData object with a .X that has been validated or transformed.
    """
        
    # subset adata.X, densify it and check if the values in it are integers 
    ds_X = adata.X[:100].copy() # NOTE: consider making this a random sampling of the matrix rather than just the first 100 rows
    ds_arr = ds_X.toarray() if sp.issparse(ds_X) else ds_X
    is_integer = np.allclose(ds_arr, np.round(ds_arr), rtol=1e-5, atol=1e-5)
    max_val = ds_arr.max()

    ## conduct heuristic and metadata checks
    if 'log1p' in adata.uns:
        print("Metadata Check: Found 'log1p' in adata.uns. Data is already transformed.")
    elif (not is_integer) and max_val < 30: # if matrix is not raw counts and max value less than 30, then it is already transformed
        print(f"Heuristic Check: 'log1p' metadata missing, but data appears transformed (contains floats, max={max_val:.2f}). Skipping.")
    else:
        print(f"Heuristic Check: Data appears not to be transformed (contains ints)")
        print('Transforming data...')
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    return adata

### clusterSize balancing related functions. These might get put into a utils directory.
def meet_target(group: pd.DataFrame, n_obs_to_keep: int, seed: int):
    # SCOPE: GLOBAL
    """_summary_

    Args:
        group (pd.DataFrame): _description_
        n_obs_to_keep (int): _description_
        seed (int): _description_

    Returns:
        _type_: _description_
    """
    if len(group) == 0:
        return group
    elif len(group) < n_obs_to_keep:
        return group.sample(n=n_obs_to_keep, replace=True, random_state=seed)
    else:
        return group.sample(n=n_obs_to_keep, replace=False, random_state=seed)

def standard_downsample(group, n_obs_to_keep: int, seed: int):
    """_summary_

    Args:
        group (_type_): _description_
        n_obs_to_keep (int): _description_
        seed (int): _description_

    Returns:
        _type_: _description_
    """
    if len(group) > n_obs_to_keep:
        return group.sample(n=n_obs_to_keep, replace=False, random_state=seed)
    else:
        return group

def balance_clusterSizes(adata: ad.AnnData, cluster_header: str, balance_groups: bool, cluster_labels: list, n_cells_to_keep: int = None, 
    standard_ds: bool = None, meet_at_value: bool = None, seed: int = seed):
    # SCOPE: GLOBAL (potentially)
    """_summary_

    Args:
        adata (ad.AnnData): _description_
        cluster_header (str): _description_
        balance_groups (bool): _description_
        cluster_labels (list): _description_
        n_cells_to_keep (int, optional): _description_. Defaults to None.
        standard_ds (bool, optional): _description_. Defaults to None.
        meet_at_value (bool, optional): _description_. Defaults to None.
        seed (int, optional): _description_. Defaults to seed.

    Returns:
        _type_: _description_
    """
    
    # wont look pretty but I'm gonna write a nasty nested conditional and even though it won't be the most readable, but I think logically, it is the best option here for 
    # compartmentalizing different functionality for different use cases
    
    # TODO:
    # [ ] implement logic for automatically calculating n_cells_to_keep
    # [ ] implement logic for doing group balancing across all cell types

    if balance_groups:
        if not n_cells_to_keep: # if user does not pass in a value, then automatically determine n_cells_to_keep
            if meet_at_value: n_cells_to_keep = "" # TODO find n_cells_to_keep by getting lowest n cluster in cluster_header, take that number, x1.5
            elif standard_ds: n_cells_to_keep = "" # TODO find n_cells_to_keep by getting lowest n cluster in cluster_header
        if cluster_labels: # if we only do bal
            print(f"Balancing clusterSizes amongst selected `cluster_labels` in adata.obs[`{cluster_header}`]")
            subset = adata.obs[adata.obs[cluster_header].isin(cluster_labels)]
            non_endo_indices = adata.obs[~adata.obs[cluster_header].isin(cluster_labels)].index.tolist()
            if meet_at_value:
                print(f"`meet_at_value` sampling strategy. Clusters in adata.obs[`{cluster_header}`] > {n_cells_to_keep} downsampled and clusters < {n_cells_to_keep} upsampled.")
                sampled_endo = subset.groupby(cluster_header, observed=True, group_keys=False).apply(meet_target, 
                                                                                                        n_obs_to_keep = n_cells_to_keep, 
                                                                                                        seed = seed)
                sampled_endo_indices = sampled_endo.index.tolist()
                all_kept_idx = sampled_endo_indices + non_endo_indices
                adata = adata[all_kept_idx].copy()
                adata.obs_names_make_unique() # gotta make obs_names unique since cells are being duplicated
            elif standard_ds:
                print(f"`standard_ds` sampling strategy. Clusters in adata.obs[`{cluster_header}`] > {n_cells_to_keep} downsampled.")
                sampled_endo = subset.groupby(cluster_header, observed=True, group_keys=False).apply(standard_downsample,
                                                                                                     n_obs_to_keep = n_cells_to_keep,
                                                                                                     seed = seed)
                sampled_endo_indices = sampled_endo.index.tolist()
                all_kept_set = set(sampled_endo_indices + non_endo_indices)        
                ordered_kept_idx = [idx for idx in adata.obs_names if idx in all_kept_set] # ensure we maintain the original order of the matrix
                adata = adata[ordered_kept_idx].copy()
            else: # error handling for missing param, this should kill this whole script
                print("Please specify a group balancing strategy: `meet_at_value` or `standard_ds`")
                sampled_endo_indices = subset.index.tolist()
                
        else: 
            print(f"Balancing clusterSizes amongst all clusters in adata.obs[`{cluster_header}`]")
            if meet_at_value:
                grouped_ad_idx = adata.obs[cluster_header].groupby(cluster_header, observed = True, group_keys=False).apply(meet_target,
                                                                                                                            n_obs_to_keep = n_cells_to_keep,
                                                                                                                            seed = seed).index.tolist()
                all_kept_set = set(grouped_ad_idx)        
                adata = adata[all_kept_set].copy()
            else:
                grouped_ad_idx = adata.obs[cluster_header].groupby(cluster_header, observed = True, group_keys=False).apply(standard_downsample,
                                                                                                                         n_obs_to_keep = n_cells_to_keep,
                                                                                                                         seed = seed).index.tolist()
                all_kept_set = set(grouped_ad_idx)        
                ordered_kept_idx = [idx for idx in adata.obs_names if idx in all_kept_set] # ensure we maintain the original order of the matrix
                adata = adata[ordered_kept_idx].copy()
    else:
        print("`balance_groups` set to `False`. adata object remains unchanged")
        return adata
    
def check_dimreds(adata: ad.AnnData, seed: int = seed):
    # SCOPE: GLOBAL
    """_summary_

    Args:
        adata (ad.AnnData): _description_
        seed (int, optional): _description_. Defaults to seed.

    Returns:
        _type_: _description_
    """
    if ("X_pca" in adata.obsm) and (adata.obsm['X_pca'].shape[1] > 30):
        print(f"adata contains viable PCA embedding with > 30 PCs")
        adata.obsm['X_pca'] = adata.obsm['X_pca'].astype(np.float64) # Formatting PCA matrix to float64 to prevent downstream bugs
    elif ("pca" in adata.obsm) and (adata.obsm['pca'].shape[1] > 30):
        print(f"adata contains viable PCA embedding with > 30 PCs")
        adata.obsm['pca'] = adata.obsm['pca'].astype(np.float64) # Formatting PCA matrix to float64 to prevent downstream bugs
    else:
        print("No `X_pca`(scanpy) or `pca`(Seurat) in adata.obsm, calculating...")
        sc.pp.pca(adata, n_comps=30, random_state=seed)
        adata.obsm['X_pca'] = adata.obsm['X_pca'].astype(np.float64)
           
    dim_red = None
    
    # check to see if any viable non-linear dim_red embeddings are present
    if ("X_umap" in adata.obsm) and (adata.obsm['X_umap'].shape[1] >= 2):
        # give priority to UMAP embeddings, so if an adata obj has this embedding, then plot it, even if it could have both embeddings
        print(f"adata contains viable UMAP embedding.")
        dim_red = "X_umap"
    elif ("X_tSNE" in adata.obsm) and (adata.obsm['X_tSNE'].shape[1] >= 2):
        print(f"adata contains viable tSNE embedding.")
        dim_red = "X_tSNE"
    else:
        print("Non-linear embedding (UMAP or tSNE) not found in adata.obsm. Calculating UMAP...")
        if "connectivities" not in adata.obsp:
            print(f"Found no `connectivities` matrix in  adata.obsp. Calculating neighborhood graph...")
            sc.pp.neighbors(adata, n_pcs=30, random_state=seed)
        
        sc.tl.umap(adata, n_components=30, random_state=seed)
        dim_red = "X_umap"
    
    return adata, dim_red


def process_h5ad(data_id, data_path, 
                 args # still don't know if args are gonna be passed here
                 ):
    # SCOPE: THIS PROJ
    """_summary_

    Args:
        data_id (_type_): _description_
    """
    print(f"\nStarting ingestion of {data_id} from {data_path}")
    
    try:
        adata = sc.read_h5ad(data_path)
    except Exception as e:
        print(f"{data_id} Error reading data: {e}")
        return
    
    ## DEALS WITH adata.var_names
    adata = resolve_var_names(adata, data_id, args.cxg, args.var_col) # not sure if argparse is still gonna be used when implementing nextflow layers

    ## HANDLING MISSING ANNOTATIONS
    print(f"Cleaning missing annotations in {args.cluster_header}...")
    adata.obs[args.cluster_header] = adata.obs[args.cluster_header].astype(object).fillna("Unknown").astype(str).astype('category')
    
    # cluster_labels arg converted from string to list
    args.cluster_labels = [str(cluster_label) for cluster_label in args.cluster_labels.split(",")]
    
    validate_cluster_labels(adata=adata, cluster_header=args.cluster_header, cluster_labels=args.cluster_labels)

    ## CHECK .X TO SEE IF ITS TRANSFORMED ALREADY
    adata = check_X_transformation(adata)

    ## IMPLEMENT DIFFERENT GROUP BALANCING STRATEGIES AMONGST CELL TYPE CLASS OF INTEREST
    adata = balance_clusterSizes(adata, cluster_header=args.cluster_header, 
                                 balance_groups=False # currently set to False until balance_groups logic is fixed. Also need to figure out if balancing amongst provided cluster_labels is a worthwhile implementation
                                 # or if I should only include logic for downsampling across the board. We know that proportionality of clusterSizes impact the quality of clusters, and balancing them returns more 
                                 # biologically representative results. This is convincing me that we should only contain logic for downsampling all clusters (if they exceed n_cells_to_keep). I think this is something 
                                 # else I should factor into the experiment I plan to do with comparing the impact of clusterSize balancing on quality of marker gene discovery.
                                 )
    
    adata, dim_red = check_dimreds(adata=adata, seed=seed)
    
    if dim_red:
        os.makedirs(os.path.join(args.results_dir, "figures", "embeddings"), exist_ok=True)
        sc.settings.figdir = os.path.join(args.results_dir, "figures", "embeddings")    
        sc.set_figure_params(dpi=200)
        sc.pl.embedding(
            adata,
            basis = dim_red,
            color = args.cluster_header,
            frameon = True,
            use_raw = False,
            save = f"_{data_id}_global_data.png"
        )
        
        sc.pl.embedding(
            adata[adata.obs[args.cluster_header].isin(args.cluster_labels)],
            basis = dim_red,
            color = args.cluster_header,
            frameon = True,
            use_raw = False,
            save = f"_{data_id}_local_data.png"
        )

    os.makedirs(os.path.join(args.tmpdir, f'{data_id}_tmp_files', 'h5ads'), exist_ok=True)
    adata.write(os.path.join(args.tmpdir, f'{data_id}_tmp_files', 'h5ads', f'{data_id}_ingested.h5ad'))
    del adata 
    gc.collect()
    
def main():
    
    parser = argparse.ArgumentParser(description="Ingest scRNA-seq data (Single Sample or Batch via Sample Sheet)")

    # parser.add_argument("--sample_sheet", type=str, default=None, help="Path to CSV sample sheet containing 'data_id' and 'data_path' columns.")

    parser.add_argument("--data_id", type=str, help="String to ID the data. Required if not using --sample_sheet.")
    parser.add_argument("--data_path", type=str, help="Path to input h5ad file. Required if not using --sample_sheet.")

    parser.add_argument("--results_dir", type=str, required=True, help="Path to save results. Directory named after --data_id.")
    parser.add_argument("--cluster_header", type=str, required=True, help="Column name of adata.obs that contains cell type labels of interest")
    parser.add_argument("--tmpdir", type=str, required=True, help="Temporary space for holding intermediate files. On Biowulf, set $TMPDIR to lscratch space.")
    parser.add_argument("--cxg", action="store_true", help="Indicate whether or not data is sourced from CellxGene. Omit if data not from CellxGene. This is to deal with how CellxGene organizes their adata.var")
    parser.add_argument("--var_col", type=str, default="", help="Column in adata.var where gene symbols are held")
    parser.add_argument("--cluster_labels", type=str, required=True, help="Comma-separated string of cluster labels that compose your local data of interest. Could represent a lineage/compartment or a certain biologically relevant grouping of cells.")
    # parser.add_argument("--balance_groups", action="store_true", help="Whether or not to balance group sizes of endothelial cells to the lowest represented group")
    # parser.add_argument("--meet_at_value", action="store_true")
    # parser.add_argument("--standard_ds", action="store_true")
    # parser.add_argument("--n_cells_to_keep", type=int, default=None, help="Target number of cells per cluster")

    args = parser.parse_args()

    process_h5ad(args.data_id, args.data_path, args)
###### FUNCTION DEFINITIONS END ######

if __name__ == "__main__":
    main()