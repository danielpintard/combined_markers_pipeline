## CONFIG
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

seed = 42 

def process_h5ad(data_id, data_path, args):
    
    print(f"\nStarting ingestion of {data_id} from {data_path}")
    
    try:
        adata = sc.read_h5ad(data_path)
    except Exception as e:
        print(f"{data_id} Error reading data: {e}")
        return
    
    if args.cxg:
        adata.var['ensembl_id'] = adata.var_names
        adata.var.index = adata.var['feature_name'].astype(str)
        adata.var_names_make_unique()
        adata.var.index.name = None
    elif args.var_col == "":
        var_col = adata.var_names
    else:
        var_col = args.var_col
        adata.var.index = adata.var[var_col].astype(str)

    ## HANDLING MISSING ANNOTATIONS
    print(f"Cleaning missing annotations in {args.cluster_header}...")
    adata.obs[args.cluster_header] = adata.obs[args.cluster_header].astype(object).fillna("Unknown").astype(str).astype('category')

    # NOTE: UPDATED CODE FOR CHECKING IF adata.X HAS BEEN TRANFORMED ALREADY
    ## CHECK .X TO SEE IF ITS TRANSFORMED ALREADY
    ds_X = adata.X[:100].copy()
    ds_arr = ds_X.toarray() if sp.issparse(ds_X) else ds_X
    is_integer = np.allclose(ds_arr, np.round(ds_arr), rtol=1e-5, atol=1e-5)
    max_val = ds_arr.max()

    if 'log1p' in adata.uns:
        print("Metadata Check: Found 'log1p' in adata.uns. Data is already transformed.")
    elif (not is_integer) and max_val < 30: # if matrix is not raw counts andmax value less than 30, then it is transformed
        print(f"Heuristic Check: 'log1p' metadata missing, but data appears transformed (contains floats, max={max_val:.2f}). Skipping.")
    else:
        print(f"Heuristic Check: Data appears not to be transformed (contains ints)")
        print('Transforming data...')
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    ## IMPLEMENT DIFFERENT GROUP BALANCING STRATEGIES AMONGST CELL TYPE CLASS OF INTEREST
    if args.balance_groups and args.n_cells_to_keep:
        subset = adata.obs[adata.obs[args.cluster_header].isin(args.endo_labels)]
        non_endo_indices = adata.obs[~adata.obs[args.cluster_header].isin(args.endo_labels)].index.tolist()
        
        if args.meet_at_value:
            def meet_target(group):
                if len(group) == 0:
                    return group
                elif len(group) < args.n_cells_to_keep:
                    return group.sample(n=args.n_cells_to_keep, replace=True, random_state=seed)
                else:
                    return group.sample(n=args.n_cells_to_keep, replace=False, random_state=seed)

            sampled_endo = subset.groupby(args.cluster_header, observed=True, group_keys=False).apply(meet_target)
            sampled_endo_indices = sampled_endo.index.tolist()

        elif args.standard_ds:
            def standard_downsample(group):
                if len(group) > args.n_cells_to_keep:
                    return group.sample(n=args.n_cells_to_keep, replace=False, random_state=seed)
                else:
                    return group
            
            sampled_endo = subset.groupby(args.cluster_header, observed=True, group_keys=False).apply(standard_downsample)
            sampled_endo_indices = sampled_endo.index.tolist()
        else:
            print("Please specify a group balancing strategy: `meet_at_value` or `standard_ds`")
            sampled_endo_indices = subset.index.tolist()
        
        if args.meet_at_value:
            # gotta make obs_names unique since cells are being duplicated
            all_kept_idx = sampled_endo_indices + non_endo_indices
            adata = adata[all_kept_idx].copy()
            adata.obs_names_make_unique()
        else:
            # ensuring we maintain order of
            all_kept_set = set(sampled_endo_indices + non_endo_indices)        
            ordered_kept_idx = [idx for idx in adata.obs_names if idx in all_kept_set]
            adata = adata[ordered_kept_idx].copy()


    # CHECK IF ADATA HAS PRECOMPUTED PCA SPACE
    if "X_pca" not in adata.obsm:
        print("No `X_pca` in .obsm, calculating...")
        sc.pp.pca(adata, n_comps=30, random_state=seed)

    os.makedirs(os.path.join(args.results_dir, "figures", "umaps"), exist_ok=True)
    sc.settings.figdir = os.path.join(args.results_dir, "figures", "umaps")    

    if ("X_umap" not in adata.obsm) and ("X_tSNE" in adata.obsm):
        #if we hae tSNE but no UMAP
        sc.pl.embedding(
            adata,
            basis = 'X_tSNE',
            color = args.cluster_header,
            frameon = False,
            use_raw = args.use_raw,
            save = f"_{data_id}_global_data_tSNE.png"
            )
        sc.pl.embedding(
            adata[adata.obs[args.cluster_header].isin(args.endo_labels)],
            basis='X_tSNE',
            color = args.cluster_header,
            frameon = False,
            use_raw = args.use_raw,
            save = f"_{data_id}_local_data_tSNE.png"
        )
    elif ("X_umap" in adata.obsm):      
        # we have UMAP and no tSNE
        sc.pl.umap(
            adata,
            color = args.cluster_header,
            frameon = False,
            use_raw = args.use_raw,
            save = f"_{data_id}_global_data.png"
        )
    else:
        # we have neither tSNE nor UMAP so make UMAP
        print(f"[{data_id}] No `X_umap` is .obsm, calculating...")
        if 'X_pca' in adata.obsm:
            print("Checking pca matrix before calculating UMAP...")
            if adata.obsm['X_pca'].shape[1] < 30:
                print("`X_pca` does not have enough PCs. Rerunning `sc.pp.pca` with 30 comps.")
                sc.pp.pca(adata, n_comps=30, random_state=seed)
        else:
            print(f"using `X_pca` with {adata.obsm['X_pca'].shape[1]} components to calculate UMAP embedding...")

        sc.pp.neighbors(adata, n_pcs=30, random_state=seed)
        sc.tl.umap(adata, n_components=30, random_state=seed)
        sc.set_figure_params(dpi=200)
        
        sc.pl.umap(
            adata,
            color = args.cluster_header,
            frameon = False,
            use_raw = args.use_raw,
            save = f"_{data_id}_global_data.png"
        )

        sc.pl.umap(
            adata[adata.obs[args.cluster_header].isin(args.endo_labels)],
            color = args.cluster_header,
            frameon = False,
            use_raw = args.use_raw,
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
    parser.add_argument("--use_raw", action="store_false", help="Flag to use adata.raw for plotting umap")
    parser.add_argument("--endo_labels", type=str, nargs='+', required=True, help="Array of endothelial labels")
    parser.add_argument("--balance_groups", action="store_true", help="Whether or not to balance group sizes of endothelial cells to the lowest represented group")
    parser.add_argument("--meet_at_value", action="store_true")
    parser.add_argument("--standard_ds", action="store_true")
    parser.add_argument("--n_cells_to_keep", type=int, default=None, help="Target number of cells per cluster")

    args = parser.parse_args()

    ## LIKELY DEPRECATED - SAMPLE SHEET READING LOGIC HANDLED BY BASH NOW 
    ## LOGIC FOR READING IN AND PROCESSING SAMPLE SHEET FOR HANDLING MULTPILE DATA SETS
    # if args.sample_sheet is None:
    #     if args.data_id is None or args.data_path is None:
    #         parser.error("You must provide either --sample_sheet OR both --data_id and --data_path")
        
    #     process_h5ad(args.data_id, args.data_path, args)
        
    # else:
    #     print(f"Reading sample sheet: {args.sample_sheet}")
    #     try:
    #         df = pd.read_csv(args.sample_sheet)
    #     except Exception as e:
    #         raise RuntimeError(f"Could not read sample sheet: {e}")
            
    #     if 'data_id' not in df.columns or 'data_path' not in df.columns:
    #         raise ValueError("Sample sheet must contain 'data_id' and 'data_path' columns.")

    #     for _, row in df.iterrows():
    #         process_h5ad(row['data_id'], row['data_path'], args)

    process_h5ad(args.data_id, args.data_path, args)
        
if __name__ == "__main__":
    main()