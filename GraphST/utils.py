import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False, resolution=None):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7. Ignored for Leiden/Louvain when resolution is provided.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'.
    start : float
        The start value for searching. The default is 0.1.
    end : float
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.
    resolution : float, optional
        If provided and method is 'leiden' or 'louvain', use this resolution directly and skip search_res.
        n_clusters is ignored in that case.

    Returns
    -------
    None.

    """
    
    pca = PCA(n_components=20, random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       if resolution is not None:
           sc.pp.neighbors(adata, n_neighbors=50, use_rep='emb_pca')
           sc.tl.leiden(adata, random_state=0, resolution=resolution)
       else:
           res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
           sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       if resolution is not None:
           sc.pp.neighbors(adata, n_neighbors=50, use_rep='emb_pca')
           sc.tl.louvain(adata, random_state=0, resolution=resolution)
       else:
           res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
           sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 
       
    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def extract_top_value(map_matrix, retain_percent = 0.1): 
    '''\
    Filter out cells with low mapping probability

    Parameters
    ----------
    map_matrix : array
        Mapped matrix with m spots and n cells.
    retain_percent : float, optional
        The percentage of cells to retain. The default is 0.1.

    Returns
    -------
    output : array
        Filtered mapped matrix.

    '''

    #retain top 1% values for each spot
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    
    return output 

def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat

def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    '''\
    Project cell types onto ST data using mapped matrix in adata.obsm

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    adata_sc : anndata
        AnnData object of scRNA-seq reference data.
    retrain_percent: float    
        The percentage of cells to retain. The default is 0.1.
    Returns
    -------
    None.

    '''
    
    # read map matrix 
    map_matrix = adata.obsm['map_matrix']   # spot x cell
   
    # extract top-k values for each spot
    map_matrix = extract_top_value(map_matrix) # filtering by spot
    
    # construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values
       
    # projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)
   
    # rename cell types
    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    #cell_type = [s.replace(' ', '_') for s in cell_type]
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)  # spot x cell type
    
    #normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    #add projection results to adata
    adata.obs[df_projection.columns] = df_projection
    
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res


def resolution_sweep_leiden(
    adata,
    start=0.1,
    end=2.0,
    step=0.05,
    use_rep="emb_pca",
    n_neighbors=50,
    random_state=0,
):
    """\
    Fix the neighbor graph once and run Leiden at many resolutions; record
    number of clusters, silhouette score, and Davies–Bouldin score per resolution.

    If adata.obsm[use_rep] is missing but 'emb' exists, PCA (20 components) is
    applied to 'emb' to create 'emb_pca'. Builds neighbors once, then sweeps
    resolution. Modifies adata in place (adds emb_pca/neighbors and overwrites
    adata.obs['leiden'] at each step).

    Parameters
    ----------
    adata : anndata.AnnData
        Must have .obsm[use_rep] or .obsm['emb'].
    start : float
        Start of resolution range (inclusive).
    end : float
        End of resolution range (exclusive).
    step : float
        Resolution step size.
    use_rep : str
        Key in adata.obsm for clustering (default 'emb_pca').
    n_neighbors : int
        For sc.pp.neighbors (default 50).
    random_state : int
        For sc.tl.leiden (default 0).

    Returns
    -------
    pd.DataFrame
        Columns: resolution, n_clusters, silhouette_score, davies_bouldin_score.
    """
    if use_rep not in adata.obsm and "emb" in adata.obsm:
        pca = PCA(n_components=20, random_state=42)
        adata.obsm[use_rep] = pca.fit_transform(adata.obsm["emb"].copy())
    X = adata.obsm[use_rep]
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    resolutions = np.arange(start, end, step)
    resolutions = np.round(resolutions, decimals=min(5, max(0, -int(np.floor(np.log10(step))))))
    rows = []
    for res in resolutions:
        res = float(res)
        sc.tl.leiden(adata, random_state=random_state, resolution=res)
        labels = adata.obs["leiden"].astype(str).values
        n_clusters = len(np.unique(labels))
        if n_clusters <= 1:
            sil = np.nan
            db = np.nan
        else:
            sil = metrics.silhouette_score(X, labels, metric="euclidean")
            db = metrics.davies_bouldin_score(X, labels)
        rows.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "silhouette_score": sil,
            "davies_bouldin_score": db,
        })
    return pd.DataFrame(rows)
