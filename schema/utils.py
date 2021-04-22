#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, copy, multiprocessing
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance, sklearn.neighbors
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict


#### local directory imports ####
oldpath = copy.copy(sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_base_config import *

sys.path = copy.copy(oldpath)
####


def get_leiden_clustering(mtx, num_neighbors=30):
        import igraph
        gNN = igraph.Graph()
        N = mtx.shape[0]
        gNN.add_vertices(N)
        schema_debug("Flag 192.30 ", mtx.shape)
        mNN = scipy.sparse.coo_matrix( sklearn.neighbors.kneighbors_graph(mtx, num_neighbors))
        schema_debug("Flag 192.40 ", mNN.shape)
        gNN.add_edges([ (i,j) for i,j in zip(mNN.row, mNN.col)])
        schema_debug("Flag 192.50 ")
        import leidenalg as la
        p1 = la.find_partition(gNN, la.ModularityVertexPartition)
        schema_debug("Flag 192.60 ")
        return   np.array([a for a in p1.membership])


    
class ScatterPlotAligner:
    """
    When seeded with a Mx2 matrix (that will be used for a scatter plot), 
        will allow you to map other Mx2 matrices so that the original scatter plot and the 
        new scatter plot will be in a similar orientation and positioning

    !!!!! DEPRECATED - do not use !!!!!
    """
    
    def __init__(self):
        """
        you will need to seed it before you can use it
        """
        self._is_seeded = False

        
    def seed(self, D):
        """
        the base matrix to be used for seeding. Should be a 2-d numpy type array. 
        IMPORTANT: a transformed version of this will be returned to you by the call. Use that for plots.

        Returns:  1) Kx2 transformed matrix and, 2) (xmin, xmax, ymin, ymax) to use with plt.xlim & plt.ylim
        """
        assert (not self._is_seeded) # Trying to seed an already-seeded ScatterPlotAligner
        assert D.shape[1] == 2

        self._is_seeded = True
        self._K = D.shape[0]
        
        mcd = sklearn.covariance.MinCovDet()
        mcd.fit(D)

        v = mcd.mahalanobis(D)
        self._valid_idx = (v < np.quantile(v,0.75)) #focus on the 75% of non-outlier data

        self._seed = D.copy()

        # translate and scale _seed so it's centered nicely around the _valid_idx's
        seed_mu = self._seed[self._valid_idx].mean(axis=0)        
        self._seed = self._seed - seed_mu
        
        pt_magnitudes = (np.sum((self._seed[self._valid_idx])**2, axis=1))**0.5
        
        d_std = pt_magnitudes.std()
        self._seed = self._seed * (2.0/d_std)  #have 1-SD pts be roughly at dist 2 from origin

        # get the bounding boxes
        u1 = 2*self._seed[self._valid_idx].min(axis=0)
        self._xmin, self._ymin = u1[0], u1[1]
        u2 = 2*self._seed[self._valid_idx].max(axis=0)
        self._xmax, self._ymax = u2[0], u2[1]

        return self._seed, (self._xmin, self._xmax, self._ymin, self._ymax)


    
    def map(self, D):
        """
        D is a 2-d numpy-type array to be mapped to the seed. require D.shape == seed.shape 

        IMPORTANT: seed() should be called first. 
        Subsequent calls to map() will transform their inputs to match the seed's orientation

        Returns:  a transformed version of D that you should use for plotting
        """
        assert self._is_seeded
        assert D.shape[0] == self._K
        assert D.shape[1] == 2
        
        dx = D.copy()

        # translate and scale dx so it's centered nicely around the _valid_idx's
        dx_mu = dx[self._valid_idx].mean(axis=0)
        dx = dx - dx_mu

        pt_magnitudes = (np.sum((dx[self._valid_idx])**2, axis=1))**0.5
        
        d_std = pt_magnitudes.std()
        dx = dx * (2.0/d_std)

        #get the rotation matrix
        # from https://igl.ethz.ch/projects/ARAP/svd_rot.pdf (pg5) and http://post.queensu.ca/~sdb2/PAPERS/PAMI-3DLS-1987.pdf

        H = np.matmul( dx[self._valid_idx].T, self._seed[self._valid_idx])
        #H = np.matmul( self._seed[self._valid_idx].T, dx[self._valid_idx])

        import numpy.linalg

        U,S,V =  numpy.linalg.svd(H, full_matrices=True)
        det_R0 = np.linalg.det(np.matmul(V, U.T))
        R1 = np.diagflat(np.ones(2))
        R1[1,1] = det_R0

        R  = np.matmul(V, np.matmul(R1, U.T))

        dx.iloc[:,:] = np.transpose( np.matmul(R, dx.T))

        return dx  #, (self._xmin, self._xmax, self._ymin, self._ymax)



def sparse_read_csv(filename, index_col=None, verbose=False):
    """
csr read of file. Does a 100-row load first to get data-types.
returns sparse matrix, rows and cols.
based on https://github.com/berenslab/rna-seq-tsne

#### Parameters

`filename`: `string`



`index_col`: `string` (default=`None`)

    name of column that serves as index_col. Passed unchanged to read_csv(...,index_col=index_col, ...)


`verbose: `boolean` (default=`False`)
    
    say stuff
    """
    
    small_chunk = pd.read_csv(filename, nrows=100)
    coltypes = dict(enumerate([a.name for a in small_chunk.dtypes.values]))

    indexL = []
    chunkL = []
    with open(filename) as file:
        for i,chunk in enumerate(pd.read_csv(filename, chunksize=3000, index_col=index_col, dtype=coltypes)):
            if verbose: print('.', end='', flush=True)
            if i==0:
                colsL = list(chunk.columns)
            indexL.extend(list(chunk.index))
            chunkL.append(sparse.csr_matrix(chunk.values.astype(float)))
            
        mat = sparse.vstack(chunkL, 'csr')
        if verbose: print(' done')
    return mat, np.array(indexL), colsL



def fast_csv_read(filename, *args, **kwargs):
    """
fast csv read. Like sparse_read_csv but returns a dense Pandas DF
    """
    
    small_chunk = pd.read_csv(filename, nrows=50)
    if small_chunk.index[0] == 0:
        coltypes = dict(enumerate([a.name for a in small_chunk.dtypes.values]))
        return pd.read_csv(filename, dtype=coltypes, *args, **kwargs)
    else:
        coltypes = dict((i+1,k) for i,k in enumerate([a.name for a in small_chunk.dtypes.values]))
        coltypes[0] = str
        return pd.read_csv(filename, index_col=0, dtype=coltypes, *args, **kwargs)


    

class SlideSeq:
    """
Utility class for Slide-Seq (Rodriques et al., Science 2019) data. Most methods are static.
    """
    
    @staticmethod
    def loadRawData(datadir, puckid, num_nmf_factors=100, prep_for_benchmarking=False):
        """
Load data for a particular puck, clean it up a bit and store as AnnData. For later use, also performs a NMF and stores those.
Borrows code from autoNMFreg_windows.py, provided with the Slide-Seq raw data.
        """
        from sklearn.preprocessing import StandardScaler
        
        puckdir = "{0}/Puck_{1}".format(datadir, puckid)
        beadmapdir = max(glob.glob("{0}/BeadMapping_*-*_????".format(puckdir)), key=os.path.getctime)
        schema_debug("Flag 314.001 ", beadmapdir)
        
        # gene exp
        gexp_file = "{0}/MappedDGEForR.csv".format(beadmapdir)
        dge = fast_csv_read(gexp_file, header = 0, index_col = 0)
        #  for faster testing runs, use below, it has just the first 500 cols of the gexp_file 
        ## dge = fast_csv_read("/tmp/a1_dge.csv", header = 0, index_col = 0)
        dge = dge.T
        dge = dge.reset_index()
        dge = dge.rename(columns={'index':'barcode'})
        schema_debug("Flag 314.010 ", dge.shape, dge.columns)
        
        # spatial location
        beadloc_file = "{0}/BeadLocationsForR.csv".format(beadmapdir)
        coords = fast_csv_read(beadloc_file, header = 0)
        coords = coords.rename(columns={'Barcodes':'barcode'})
        coords = coords.rename(columns={'barcodes':'barcode'})
        schema_debug("Flag 314.020 ", coords.shape, coords.columns)
        
        # Slide-Seq cluster assignments
        atlas_clusters_file = "{0}/AnalogizerClusterAssignments.csv".format(beadmapdir)
        clstrs = pd.read_csv(atlas_clusters_file, index_col=None)
        assert list(clstrs.columns) == ["Var1","x"]
        clstrs.columns = ["barcode","atlas_cluster"]
        clstrs = clstrs.set_index("barcode")
        schema_debug("Flag 314.030 ", clstrs.shape, clstrs.columns)
        
        df_merged = dge.merge(coords, right_on='barcode', left_on='barcode')
        df_merged = df_merged[ df_merged.barcode.isin(clstrs.index)]
        schema_debug("Flag 314.040 ", df_merged.shape, df_merged.columns)
            
        # remove sparse gene exp
        counts = df_merged.drop(['xcoord', 'ycoord'], axis=1)
        counts2 = counts.copy(deep=True)
        counts2 = counts2.set_index('barcode') #.drop('barcode',axis=1)
        counts2_okcols = counts2.sum(axis=0) > 0
        counts2 = counts2.loc[:, counts2_okcols]
        UMI_threshold = 5
        counts2_umis = counts2.sum(axis=1).values
        counts2 = counts2.loc[counts2_umis > UMI_threshold,:]
        schema_debug("Flag 314.0552 ", counts.shape, counts2.shape, counts2_umis.shape,isinstance(counts2, pd.DataFrame))
        
        #slide-seq authors normalize to have sum=1 across each bead, rather than 1e6
        cval = counts2_umis[counts2_umis>UMI_threshold]
        if not prep_for_benchmarking:
            counts2 = counts2.divide(cval, axis=0) #np.true_divide(counts2, counts2_umis[:,None])
            #counts2 = np.true_divide(counts2, counts2_umis[:,None])
                
            # this is also a little unusual, but I'm following their practice
            counts2.iloc[:,:] = StandardScaler(with_mean=False).fit_transform(counts2.values)
            schema_debug("Flag 314.0553 ", counts2.shape, counts2_umis.shape,isinstance(counts2, pd.DataFrame))
        
        coords2 = df_merged.loc[ df_merged.barcode.isin(counts2.index), ["barcode","xcoord","ycoord"]].copy(deep=True)
        coords2 = coords2.set_index('barcode') #.drop('barcode', axis=1)
        schema_debug("Flag 314.0555 ", coords2.shape,isinstance(coords2, pd.DataFrame))
        
        ok_barcodes = set(coords2.index) & set(counts2.index) & set(clstrs.index)
        schema_debug("Flag 314.060 ", coords2.shape, counts2.shape, clstrs.shape, len(ok_barcodes))
                
        if prep_for_benchmarking:
            return (counts2[counts2.index.isin(ok_barcodes)].sort_index(), coords2[coords2.index.isin(ok_barcodes)].sort_index(), clstrs[clstrs.index.isin(ok_barcodes)].sort_index())

        ## do NMF
        K1 = num_nmf_factors
        listK1 = ["P{}".format(i+1) for i in range(K1)]
        random_state = 17 #for repeatability, a fixed value
        model1 = sklearn.decomposition.NMF(n_components= K1, init='random', random_state = random_state, alpha = 0, l1_ratio = 0)
        Ho = model1.fit_transform(counts2.values)  #yes, slideseq code had Ho and Wo mixed up. Just following their lead here. 
        Wo = model1.components_

        schema_debug("Flag 314.070 ", Ho.shape, Wo.shape)
        
        Ho_norm = StandardScaler(with_mean=False).fit_transform(Ho)
        Ho_norm = pd.DataFrame(Ho_norm)
        Ho_norm.index = counts2.index
        Ho_norm.columns = listK1
        Wo = pd.DataFrame(Wo)
        Wo.index = listK1; Wo.index.name = "Factor"
        Wo.columns = list(counts2.columns)

        Ho_norm = Ho_norm[Ho_norm.index.isin(ok_barcodes)]
        Ho_norm = Ho_norm / Ho_norm.std(axis=0)

        schema_debug("Flag 314.080 ", Ho_norm.shape, Wo.shape)
        
        genexp = counts2[ counts2.index.isin(ok_barcodes)].sort_index()
        beadloc = coords2[ coords2.index.isin(ok_barcodes)].sort_index()
        clstrs = clstrs[ clstrs.index.isin(ok_barcodes)].sort_index()
        Ho_norm = Ho_norm.sort_index()

        schema_debug("Flag 314.090 ", genexp.shape, beadloc.shape, clstrs.shape, Ho_norm.shape, genexp.index[:5], beadloc.index[:5])

        beadloc["atlas_cluster"] = clstrs["atlas_cluster"]
        
        if "AnnData" not in dir():
            from anndata import AnnData

        adata = AnnData(X = genexp.values, obs = beadloc, uns = {"Ho": Ho_norm, "Ho.index": list(Ho_norm.index), "Ho.columns": list(Ho_norm.columns),
                                                                 "Wo": Wo, "Wo.index": list(Wo.index), "Wo.columns": list(Wo.columns)})
        return adata



    @staticmethod
    def loadAnnData(fpath):
        """
Import a h5ad file. Also deals with some scanpy weirdness when loading dataframes in the .uns slot
        """
        
        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 

        adata = sc.read(fpath)
        for k in ["Ho","Wo"]:
            adata.uns[k] = pd.DataFrame(adata.uns[k], index=adata.uns[k + ".index"], columns= adata.uns[k + ".columns"])
        return adata

    
    
        
class SciCar:
    """
Utility class for Sci-Car (Cao et al., Science 2018) data. Most methods are static. 
    """

    
    @staticmethod
    def loadData(path, gsm_info, refdata):
        """
Load sci-CAR data format, as uploaded to GEO. Written for DOI:10.1126/science.aau0730 
but will probably work with other Shendure Lab datasets as well

#### Parameters

`path`: `string`

    directory where the files are


`gsm_info`:  `list` of (`string`, `string`, `string`, `function`, `function`) tuples

    Each tuple corresponds to one dataset (e.g. RNA-seq or ATAC-seq)
    The list should be in order of importance. Once a feature name for a modality shows up, 
    it will be ignored in subsequent tuples
    The cells will be taken as the intersection of "sample" column from the various cell files

    tuple = (name, modality, gsm_id, f_cell_filter, f_mdlty_filter)

       name: your name for the dataset. 
       modality: 'gene' or 'peak' for now
       gsm_id: GSMXXXXXX id
       f_cell_filter: None or a function that returns a boolean given a row-vector, used to filter
       f_mdlty_filter: None or a function that returns a boolean given a row-vector, used to filter


`refdata`: `string`: path to a tab-separated file

    Read as a dataframe containing Ensemble IDs ("ensembl_id"), TSS start/end etc. 

#### Returns

AnnData object where the cells are rows in X, the columns of dataset #1 are in .X, 
    the columns for remaining datasets are in .uns dicts keyed by given dataset name.
    The cell-line info is in .obs, while dataset #1's modality info is in .var
        """

        dref = pd.read_csv(refdata, sep="\t", low_memory=False)
        dref["tss_adj"] = np.where(dref["strand"]=="+", dref["txstart"], dref["txend"])
        dref = dref["ensembl_id,chr,strand,txstart,txend,tss_adj,map_location,Symbol".split(",")]
        
        if "AnnData" not in dir():
            from anndata import AnnData
        
        datasets = []
        cell_sets = []
        for nm, typestr, gsm, f_cell_filter, f_mdlty_filter in gsm_info:
            try:
                assert typestr in ['gene','peak']
                
                cells = pd.read_csv(glob.glob("{0}/{1}*_cell.txt".format(path, gsm))[0], low_memory=False)
                mdlty = pd.read_csv(glob.glob("{0}/{1}*_{2}.txt".format(path, gsm, typestr))[0], low_memory=False)

                
                cell_idx = np.full((cells.shape[0],),True)
                if f_cell_filter is not None:
                    cell_idx = (cells.apply(f_cell_filter, axis=1, result_type="reduce")).values
                    
                if typestr=="gene":
                    mdlty_idx = np.full((mdlty.shape[0],),True)
                    if f_mdlty_filter is not None:
                        mdlty_idx = (mdlty.apply(f_mdlty_filter, axis=1, result_type="reduce")).values

                    schema_debug ("Flag 324.112 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                    mdlty["gene_id"] = mdlty["gene_id"].apply(lambda v: v.split('.')[0])
                    mdlty_idx[~(mdlty["gene_id"].isin(dref["ensembl_id"]))] = False

                    schema_debug ("Flag 324.113 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                    mdlty["index"] = np.arange(mdlty.shape[0])
                    mdlty = (pd.merge(mdlty, dref, left_on="gene_id", right_on="ensembl_id", how="left"))
                    mdlty = mdlty.drop_duplicates("index").sort_values("index").reset_index(drop=True).drop(columns=["index"])

                    schema_debug ("Flag 324.114 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))

                    def f_is_standard_chromosome(v):
                        try:
                            assert v[:3]=="chr" and ("_" not in v) and (v[3] in ['X','Y'] or int(v[3:])<=22 )
                            return True
                        except:
                            return False

                    mdlty_idx[~(mdlty["chr"].apply(f_is_standard_chromosome))] = False

                    schema_debug ("Flag 324.115 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                else:                    
                    mdlty_idx = np.full((mdlty.shape[0],),True)
                    if f_mdlty_filter is not None:
                        mdlty_idx = (mdlty.apply(f_mdlty_filter, axis=1, result_type="reduce")).values

                
                    
                # data = pd.DataFrame(data = scipy.io.mmread(glob.glob("{0}/{1}*_count.txt".format(path, gsm))[0]).T.tocsr().astype(np.float_),
                #                     index = cells['sample'],
                #                     columns = mdlty['mdlty_short_name'])

                data = scipy.io.mmread(glob.glob("{0}/{1}*_{2}_count.txt".format(path, gsm, typestr))[0]).T.tocsr().astype(np.float_)

                schema_debug ("Flag 324.12 read {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))


                schema_debug ("Flag 324.13 filtered  {0}:{1}:{2} cells, {3}:{4}:{5} mdlty".format( len(cell_idx), np.sum(cell_idx), cell_idx.shape, len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                data = data[cell_idx, :]
                data = data[:, mdlty_idx]
                #data = data[cell_idx,:] # mdlty_idx]
                cells = cells[cell_idx].reset_index(drop=True)
                mdlty = mdlty[mdlty_idx].reset_index(drop=True)

                schema_debug ("Flag 324.14 filtered down to {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))
                

                schema_debug ("Flag 324.15 filtered down to {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))
                
                sortidx = np.argsort(cells['sample'].values)
                cells = cells.iloc[sortidx,:].reset_index(drop=True)
                data = data[sortidx, :]

                schema_debug ("Flag 324.17 \n {0} \n {1}".format( cells.head(2), data[:2,:2]))
                       
                datasets.append((nm, typestr, data, cells, mdlty))
                
                cell_sets.append(set(cells['sample'].values))
                
            except:
                raise 
                #raise ValueError('{0} could not be read in {1}'.format(nm, path))

        common_cells = list(set.intersection(*cell_sets))
        schema_debug ("Flag 324.20 got {0} common cells {1}".format( len(common_cells), common_cells[:10]))

        def logcpm(dx):
            libsizes = 1e-6 + np.sum(dx, axis=1)
            dxout = copy.deepcopy(dx)
            for i in range(dxout.shape[0]):
                i0,i1 = dxout.indptr[i], dxout.indptr[i+1]
                dxout.data[i0:i1] = np.log2(dxout.data[i0:i1]*1e6/libsizes[i] + 1)
                #for ind in range(i0,i1):
                #    dxout.data[ind] = np.log2( dxout.data[ind]*1e6/libsizes[i] + 1)
            return dxout

               
        for i, dx in enumerate(datasets):
            nm, typestr, data, cells, mdlty = dx
            
            cidx = np.in1d(cells["sample"].values, common_cells)
            schema_debug ("Flag 324.205 got {0} {1} {2} {3}".format( nm, cells.shape, len(cidx), np.sum(cidx)))
                    
            data = data[cidx,:]
            cells = cells.iloc[cidx,:].reset_index(drop=True)
            mdlty = mdlty.set_index('gene_short_name' if typestr=='gene' else 'peak')
            
            if i==0:
                cells = cells.set_index('sample')
                #adata = AnnData(X = logcpm(data), obs = cells.copy(deep=True), var = mdlty.copy(deep=True))
                adata = AnnData(X = data, obs = cells.copy(deep=True), var = mdlty.copy(deep=True))
                adata.uns["names"] = []

                schema_debug ("Flag 324.22 got X {0} obs {1} var {2} uns {3}".format( adata.X.shape, adata.obs.shape, adata.var.shape, list(adata.uns.keys())))
            else:
                for c in cells.columns:
                    if c in adata.obs.columns: continue
                    adata.obs[c] = cells[c]
                #adata.uns[nm + ".X"] = logcpm(data)
                adata.uns[nm + ".X"] = data
                adata.uns[nm + ".var"] = mdlty.copy(deep=True)
                adata.uns[nm + ".var.index"] = list(mdlty.index) #scanpy is annoying, it'll convert these to numpy matrices when writing
                adata.uns[nm + ".var.columns"] = list(mdlty.columns)

            adata.obs[typestr + "_log_sumcounts"] = np.log2(np.sum(data,axis=1)+1)
            adata.uns["names"].append(nm)
            adata.uns[nm + ".type"] = typestr
            adata.var_names_make_unique()
            schema_debug ("Flag 324.25 got X {0} obs {1} var {2} uns {3}".format( adata.X.shape, adata.obs.shape, adata.var.shape, list(adata.uns.keys())))
                   
        return adata

    @staticmethod
    def loadAnnData(fpath):
        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 

        adata = sc.read(fpath)
        for k in adata.uns.keys():
            if k.endswith(".var") and isinstance(adata.uns[k], np.ndarray):
                schema_debug("Flag 343.100 hello", k, adata.uns.keys(), type(adata.uns[k]))
                adata.uns[k] = pd.DataFrame(adata.uns[k], index= adata.uns[k + ".index"], columns = adata.uns[k + ".columns"])
                schema_debug("Flag 343.102 ", k, adata.uns.keys(), type(adata.uns[k]))
                for c in  adata.uns[k].columns:
                    if c in ["id","start","end"]:
                        adata.uns[k][c] = adata.uns[k][c].astype(int)
                    if c in ["chr"]:
                        adata.uns[k][c] = adata.uns[k][c].astype(str)
        return adata

    @staticmethod
    def preprocessAnnData(adata, do_logcpm=True, valid_gene_minobs=0, valid_peak_minobs=0, valid_cell_mingenes=0):
        """
Preprocess sci-CAR data to remove too-sparse genes, peaks and cells. Also, convert to log(counts_per_million(..)) format

#### Parameters

`adata`: `AnnData`

    The dataframe containing Read as a dataframe containing Ensemble IDs ("ensembl_id"), TSS start/end etc. 


`do_logcpm`: `bool` 

    Convert peak and gene expression counts to log2 counts-per-million


`valid_gene_minobs`: `int`

    Only keep genes that show up in at least valid_gene_minobs cells


`valid_aux_minobs`: `int`

    Only keep peaks that show up in at least valid_peak_minobs cells


`valid_cell_mingenes`: `int`

    Only keep cells that have at least valid_cell_mingenes genes

#### Returns

copy of filtered anndata
         """

        valid_genes =  np.ravel((adata.X  > 0).sum(axis=0)) >= valid_gene_minobs
        adata = adata[:, valid_genes]
        
        valid_cells = np.ravel((adata.X > 0).sum(axis=1)) >= valid_cell_mingenes
        adata = adata[valid_cells, :]
        
        if "atac.X" in adata.uns:
            adata.uns["atac.X"] = adata.uns["atac.X"][ valid_cells, :]
            
            valid_peaks = np.ravel((adata.uns["atac.X"] > 0).sum(axis=0)) >= valid_peak_minobs
            adata.uns["atac.X"] = adata.uns["atac.X"][:, valid_peaks]
            adata.uns["atac.var"] = adata.uns["atac.var"][valid_peaks]
            adata.uns["atac.var.index"] = adata.uns["atac.var.index"][valid_peaks]
            
        adata2 = adata.copy()
         
        def logcpm(dx):
            libsizes = 1e-6 + np.sum(dx, axis=1)
            schema_debug ("Flag 3343.10 ",  libsizes.shape, libsizes.sum())
            dxout = dx #copy.deepcopy(dx)
            for i in range(dxout.shape[0]):
                i0,i1 = dxout.indptr[i], dxout.indptr[i+1]
                dxout.data[i0:i1] = np.log2(dxout.data[i0:i1]*1e6/libsizes[i] + 1)
                #for ind in range(i0,i1):
                #    dxout.data[ind] = np.log2( dxout.data[ind]*1e6/libsizes[i] + 1)
            return dxout
                 
        adata2.X = logcpm(adata2.X)
        if  "atac.X" in adata.uns:
            adata2.uns["atac.X"] = logcpm(adata2.uns["atac.X"])

        return adata2


    
    @staticmethod
    def getChrMapping(adata):    
        """
Get a mapping of genes/peaks to chromosomes and back

#### Parameters

`adata`: `AnnData` object

    output from loadData(...)

#### Returns
    gchr, chr2genes, pchr, chr2peaks : the first and third are integer vectors, the second and fourth are int->set(int) dicts. All gene and peak
      integers  refer to indexes 
        """
        chr2genes = defaultdict(set)
        chr2peaks = defaultdict(set)
        gchr = adata.var["chr"].astype(str).apply(lambda s: s.replace("chr",""))
        for i in range(adata.var.shape[0]):
            chr2genes[gchr[i]].add(i)
        pchr = adata.uns["atac.var"]["chr"].astype(str).apply(lambda s: s.replace("chr",""))
        for i in range(adata.uns["atac.var"].shape[0]):
            chr2peaks[pchr[i]].add(i)
        return gchr, chr2genes, pchr, chr2peaks


    @staticmethod
    def getPeakPosReGenes(gVar, peak, gene2chr):
        """
Given a peak, get its position vis-a-vis the genes in the genome

#### Parameters

`gVar`:  `pd.DataFrame`

    adata.var df from adata object


`peak`: `list` of size 3

   [chr, start, end] identifying the peak

`gene2chr`: `array of strings`

   for each gene idx, indicates which chromosome it is a part of 
 
#### Returns
    gVar.shape[0] x 5 nd-array, with 5 numbers describing the peak's posn re the gene (see code)
        """
        pchr, pstart, pend = peak
        schema_debug ("Flag 321.10012 ", peak)
        pchr = (str(pchr)).replace("chr","")


        #5 dims: 0: same chr, 1: pStart-gStart, 2: pEnd-gStart, 3: pStart-gEnd, 4: pEnd-gEnd
        pos = np.zeros((gVar.shape[0],5))
        pos[:,0] = np.where(gene2chr==pchr,1,0)
        pos[:,1] = np.where(gVar["strand"]=="+", gVar["txstart"]-pstart, pend-gVar["txend"]) #+ if pstart is upstream of txstart
        pos[:,2] = np.where(gVar["strand"]=="+", gVar["txstart"]-pend, pstart-gVar["txend"]) #+ if pend is upstream of txstart
        pos[:,3] = np.where(gVar["strand"]=="+", gVar["txend"]-pstart, pend-gVar["txstart"]) #+ if pstart is upstream of txend
        pos[:,4] = np.where(gVar["strand"]=="+", gVar["txend"]-pend, pstart-gVar["txstart"]) #+ if pend is upstream of txend

        assert np.sum((pos[:,2] > 0) & (pos[:,1] <= 0)) ==0
        assert np.sum((pos[:,1] > 0) & (pos[:,3] <= 0)) ==0
        assert np.sum((pos[:,3] < 0) & (pos[:,4] >= 0)) ==0
        assert np.sum((pos[:,4] < 0) & (pos[:,2] >= 0)) ==0

        # pos[:,1] = np.where(gVar["strand"]=="+", pstart-gVar["txstart"], gVar["txend"]-pend)
        # pos[:,2] = np.where(gVar["strand"]=="+", pend-gVar["txstart"], gVar["txend"]-pstart)
        # pos[:,3] = np.where(gVar["strand"]=="+", pstart-gVar["txend"], gVar["txstart"]-pend)
        # pos[:,4] = np.where(gVar["strand"]=="+", pend-gVar["txend"], gVar["txstart"]-pstart)
        return pos


    @staticmethod
    def fpeak_0_500(pos):
        """
        peak ends within 500bp upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>=0) & 
                         (pos[:,2] < 5e2)), 1, 0)

    @staticmethod
    def fpeak_500_2e3(pos):
        """
        peak ends within 500-2000bp upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 5e2) & 
                         (pos[:,2] <= 2e3)), 1, 0)

    @staticmethod
    def fpeak_2e3_20e3(pos):
        """
        peak ends within 2k-20kb  upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 2e3) & 
                         (pos[:,2] <= 20e3)), 1, 0)

    @staticmethod
    def fpeak_20e3_100e3(pos):
        """
        peak ends within 20-100kb upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 20e3) & 
                         (pos[:,2] <= 100e3)), 1, 0)


    @staticmethod
    def fpeak_100e3_1e6(pos):
        """
        peak ends within 100kb-1Mb upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 100e3) & 
                         (pos[:,2] <= 1e6)), 1, 0)

    @staticmethod
    def fpeak_1e6_10e6(pos):
        """
        peak ends within 1Mb-10Mb upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 1e6) & 
                         (pos[:,2] <= 10e6)), 1, 0)


    @staticmethod
    def fpeak_10e6_20e6(pos):
        """
        peak ends within 10Mb-20Mb upstream of gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,2]>0) & 
                         (pos[:,2]> 10e6) & 
                         (pos[:,2] <= 20e6)), 1, 0)


    @staticmethod
    def fpeak_crossing_in(pos):
        """
        peak spans the TSS of the gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,1]>0) & 
                         (pos[:,2]<=0) &
                         (pos[:,4]>0)), 1, 0)


    @staticmethod
    def fpeak_inside(pos):
        """
        peak is between the start and end of the gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,1]<0) & 
                         (pos[:,2]<0) & 
                         (pos[:,3]>0) & 
                         (pos[:,4]>0)), 1, 0)


    @staticmethod
    def fpeak_crossing_out(pos):
        """
        peak is spands the txend of the gene
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,1]<0) & 
                         (pos[:,2]<0) & 
                         (pos[:,3]>0) & 
                         (pos[:,4]<0)), 1, 0)



    @staticmethod
    def fpeak_behind_1e3(pos):
        """
        peak starts within 1kb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] < 1e3)), 1, 0)


    @staticmethod
    def fpeak_behind_1e3_20e3(pos):
        """
        peak starts within 1-20kb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] > 1e3) &
                         (-pos[:,3] < 20e3)), 1, 0)


    @staticmethod
    def fpeak_behind_20e3_100e3(pos):
        """
        peak starts within 20kb-100kb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] > 20e3) &
                         (-pos[:,3] < 100e3)), 1, 0)

    @staticmethod
    def fpeak_behind_100e3_1e6(pos):
        """
        peak starts within 100kb-1Mb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] > 100e3) &
                         (-pos[:,3] < 1e6)), 1, 0)


    @staticmethod
    def fpeak_behind_1e6_10e6(pos):
        """
        peak starts within 1Mb-10Mb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] > 1e6) &
                         (-pos[:,3] < 10e6)), 1, 0)

    
    @staticmethod
    def fpeak_behind_10e6_20e6(pos):
        """
        peak starts within 10Mb-20Mb of gene end
        """
        return np.where(((pos[:,0]> 0) & 
                         (pos[:,3]<0) & 
                         (-pos[:,3] > 10e6) &
                         (-pos[:,3] < 20e6)), 1, 0)

    
    @staticmethod
    def fpeak_rbf_500(pos):
        """
        exp(-d/500)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/5e2)**2)), 0)


    @staticmethod
    def fpeak_rbf_1e3(pos):
        """
        exp(-d/1e3)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/1e3)**2)), 0)


    @staticmethod
    def fpeak_rbf_5e3(pos):
        """
        exp(-d/5e3)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/5e3)**2)), 0)


    @staticmethod
    def fpeak_rbf_20e3(pos):
        """
        exp(-d/20e3)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/20e3)**2)), 0)


    @staticmethod
    def fpeak_rbf_100e3(pos):
        """
        exp(-d/100e3)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/100e3)**2)), 0)


    @staticmethod
    def fpeak_rbf_1e6(pos):
        """
        exp(-d/1e6)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/1e6)**2)), 0)


    @staticmethod
    def fpeak_rbf_10e6(pos):
        """
        exp(-d/10e6)^2 if peak ends upstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,2]>=0), 
                                  np.exp(-((pos[:,2]/10e6)**2)), 0)


    @staticmethod
    def fpeak_behind_rbf_20e3(pos):
        """
        exp(-d/20e3)^2 if peak starts downstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,3]>=0), 
                                  np.exp(-((pos[:,3]/20e3)**2)), 0)


    @staticmethod
    def fpeak_behind_rbf_100e3(pos):
        """
        exp(-d/100e3)^2 if peak starts downstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,3]>=0), 
                                  np.exp(-((pos[:,3]/100e3)**2)), 0)


    @staticmethod
    def fpeak_behind_rbf_1e6(pos):
        """
        exp(-d/1e6)^2 if peak starts downstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,3]>=0), 
                                  np.exp(-((pos[:,3]/1e6)**2)), 0)


    @staticmethod
    def fpeak_behind_rbf_10e6(pos):
        """
        exp(-d/10e6)^2 if peak starts downstream of gene
        """
        return np.where((pos[:,0]> 0) & (pos[:,3]>=0), 
                                  np.exp(-((pos[:,3]/10e6)**2)), 0)
    


    fpeak_list_all = [ (fpeak_0_500.__func__, "fpeak_0_500"),
                       (fpeak_500_2e3.__func__, "fpeak_500_2e3"),
                       (fpeak_2e3_20e3.__func__, "fpeak_2e3_20e3"),
                       (fpeak_20e3_100e3.__func__, "fpeak_20e3_100e3"),
                       (fpeak_100e3_1e6.__func__, "fpeak_100e3_1e6"),
                       (fpeak_1e6_10e6.__func__, "fpeak_1e6_10e6"),
                       (fpeak_10e6_20e6.__func__, "fpeak_10e6_20e6"),
                       (fpeak_crossing_in.__func__, "fpeak_crossing_in"),
                       (fpeak_inside.__func__, "fpeak_inside"),
                       (fpeak_crossing_out.__func__, "fpeak_crossing_out"),
                       (fpeak_behind_1e3.__func__, "fpeak_behind_1e3"),
                       (fpeak_behind_1e3_20e3.__func__, "fpeak_behind_1e3_20e3"),
                       (fpeak_behind_20e3_100e3.__func__, "fpeak_behind_20e3_100e3"),
                       (fpeak_behind_100e3_1e6.__func__, "fpeak_behind_100e3_1e6"),
                       (fpeak_behind_1e6_10e6.__func__, "fpeak_behind_1e6_10e6"),
                       (fpeak_behind_10e6_20e6.__func__, "fpeak_behind_10e6_20e6"),
                       (fpeak_rbf_500.__func__, "fpeak_rbf_500"),
                       (fpeak_rbf_1e3.__func__, "fpeak_rbf_1e3"),
                       (fpeak_rbf_5e3.__func__, "fpeak_rbf_5e3"),
                       (fpeak_rbf_20e3.__func__, "fpeak_rbf_20e3"),
                       (fpeak_rbf_100e3.__func__, "fpeak_rbf_100e3"),
                       (fpeak_rbf_1e6.__func__, "fpeak_rbf_1e6"),
                       (fpeak_rbf_10e6.__func__, "fpeak_rbf_10e6"),
                       (fpeak_behind_rbf_20e3.__func__, "fpeak_behind_rbf_20e3"),
                       (fpeak_behind_rbf_100e3.__func__, "fpeak_behind_rbf_100e3"),
                       (fpeak_behind_rbf_1e6.__func__, "fpeak_behind_rbf_1e6"),
                       (fpeak_behind_rbf_10e6.__func__, "fpeak_behind_rbf_10e6"),
    ]
    


    @staticmethod
    def computeGeneByFpeakMatrix(adata, peak_func_list, chr_mapping = None, peakList = None, normalize_distwt=True, booleanPeakCounts = False):
        """
Compute a matrix that is nG x len(peak_func_list) with cell [i,j] being gene[i]'s dot-product with peak scores across all peaks, 
         subject to  peak wt as described by f_peak

#### Parameters

`adata` : AnnData object from loadData(...)

`peak_func_list` : `list` of  functions of the signature `pos: int`

`chr_mapping`:  `4-tuple` 

    Optional argument providing the output of `getChrMapping(..)`, for caching

`peakList`:  `list of int`

   Optional argument specifying which peak indexes to run over 

#### Returns
 
a   matrix of shape adata.var.shape[0] X len(peak_func_list)

        """

        if chr_mapping is None:
            gene2chr, chr2genes, peak2chr, chr2peaks = SciCar.getChrMapping(adata)
        else:
            gene2chr, chr2genes, peak2chr, chr2peaks = chr_mapping

        nCells, nGenes, nPeaks = adata.shape[0], adata.shape[1], adata.uns["atac.X"].shape[1]
        try:
            gXt = adata.X.T.tocsr()
        except:
            gXt = adata.X.T

        if booleanPeakCounts and gXt.shape[0] > 0:
            gXt = gXt > np.median(gXt, axis=0) ## UNDO ????

        pXt = adata.uns["atac.X"].T.tocsr()
        gVar = adata.var[["chr","strand","txstart","txend","tss_adj"]]
        pVar = adata.uns["atac.var"][["chr","start","end"]]

        k = len(peak_func_list)
        
        g2p = np.zeros((nGenes,k))
        g2p_wts = np.zeros((nGenes,k))

        if peakList is not None:
            plist = peakList
        else:
            plist = range(nPeaks)

        for i in plist:
            v = SciCar.getPeakPosReGenes(gVar, pVar.values[i,:], gene2chr)

            pXti =  np.ravel(pXt[i,:].todense())

            pXti_positive = (pXti > 0).astype(int)
            pXti_ones = np.ones_like(pXti)

            schema_debug ("Flag 2.0008 ",i, flush=True)

            for j,f in enumerate(peak_func_list):
                distwt = f(v)
                #schema_debug("Flag 2.0010 ", i,j,np.std(distwt), np.mean(distwt)) 
                if np.sum(distwt)>1e-12:
                    if normalize_distwt: distwt = distwt/np.mean(distwt)
                    G = gXt.copy()
                    try:
                        G.data *= distwt.repeat(np.diff(G.indptr))
                    except:
                        G = G*distwt[:,None]
                    # schema_debug ("Flag 2.0201 ", len(distwt), np.sum(G.data), G.shape, 
                    #        np.sum(gXt.data),
                    #        G.data.shape, len(np.diff(G.indptr)),
                    #       pXt.shape)
                    gw = G.dot(pXti).ravel()
                    if booleanPeakCounts:
                        gw = G.dot(pXti_positive).ravel()
                    g2p[:,j] += gw/nCells

                    gw_ones = G.dot(pXti_ones).ravel()
                    g2p_wts[:,j] += gw_ones/nCells

                    # schema_debug ("Flag 2.0320 ", G.shape, distwt.shape, gXt.shape, pXt.shape, gw.shape, g2p.shape)
                
        return (g2p, g2p_wts)


    


    @staticmethod
    def aggregatePeaksByGenes(adata, peak_func, chr_mapping = None):
        """
Compute a matrix that is nCells x nGenes with cell [i,j] being expression of peaks around gene[i] in cell [i], 
         subject to  peak wt as described by f_peak

#### Parameters

`adata` : AnnData object from loadData(...)

`peak_func` : function of the signature `pos: int`

`chr_mapping`:  `4-tuple` 

    Optional argument providing the output of `getChrMapping(..)`, for caching

#### Returns
 
a   matrix of shape adata.shape

        """
        
        
        if chr_mapping is None:
            gene2chr, chr2genes, peak2chr, chr2peaks = SciCar.getChrMapping(adata)
        else:
            gene2chr, chr2genes, peak2chr, chr2peaks = chr_mapping

        nCells, nGenes, nPeaks = adata.shape[0], adata.shape[1], adata.uns["atac.X"].shape[1]

        pX = adata.uns["atac.X"]

        exp_pX = np.exp(pX)  #data was log1p'd before. We'll add the cpm counts and then re-log1p it 
        gVar = adata.var[["chr","strand","txstart","txend","tss_adj"]]
        pVar = adata.uns["atac.var"][["chr","start","end"]]

        c2g = np.zeros(nCells, nGenes)
        c2g_tmp = np.zeros(nCells, nGenes)
        
        plist = range(nPeaks)

        for i in plist:
            c2g_tmp[:] = 0
            
            v = SciCar.getPeakPosReGenes(gVar, pVar.values[i,:], gene2chr)
            c2g_tmp += peak_func(v)
            p_i =  np.exp(np.ravel(pX[i,:].todense()))
            c2g_tmp *= p_i[:,None]

            c2g += c2g_tmp

        c2g = np.log1p(c2g)
        return c2g

    





def cmpSpearmanVsPearson(d1, type1="numeric", d2=None, type2=None, nPointPairs=2000000, nRuns=1):

    N, K1 = d1.shape[0], d1.shape[1]
    K2 = 0
    if d2 is not None:
        assert d2.shape[0] == N
        K2 = d2.shape[1]
        
    if nPointPairs is None: assert nRuns==1

    schema_debug ("Flag 676.10 ", N, K1, K2, nPointPairs, type1, type2)
    
    corrL = []

    for nr in range(nRuns):
        if nPointPairs is not None:
            j_u = np.random.randint(0, N, int(3*nPointPairs))
            j_v = np.random.randint(0, N, int(3*nPointPairs))
            valid = j_u < j_v  #get rid of potential duplicates (x1,x2) and (x2,x1) as well as (x1,x1)
            i_u = (j_u[valid])[:nPointPairs]
            i_v = (j_v[valid])[:nPointPairs]
        else:
            x = pd.DataFrame.from_records(list(itertools.combinations(range(N),2)), columns=["u","v"])
            x = x[x.u < x.v]
            i_u = x.u.values
            i_v = x.v.values

        schema_debug ("Flag 676.30 ", i_u.shape, i_v.shape, nr)

        dL = []
        for g_val, g_type in [(d1, type1), (d2, type2)]:
            if g_val is None:
                dL.append(None)
                continue

            dg = []
            for ii in np.split(np.arange(i_u.shape[0]), int(i_u.shape[0]/2000)):
                ii_u = i_u[ii]
                ii_v = i_v[ii]

                if g_type == "categorical":
                    dgx = 1.0*( g_val[ii_u] != g_val[ii_v]) #1.0*( g_val[i_u].toarray() != g_val[i_v].toarray())
                elif g_type == "feature_vector":
                    schema_debug (g_val[ii_u].shape, g_val[ii_v].shape)
                    dgx = np.ravel(np.sum(np.power(g_val[ii_u].astype(np.float64) - g_val[ii_v].astype(np.float64),2), axis=1))
                else:  #numeric
                    dgx = (g_val[ii_u].astype(np.float64) - g_val[ii_v].astype(np.float64))**2   #(g_val[i_u].toarray() - g_val[i_v].toarray
                dg.extend(dgx)
            schema_debug ("Flag 676.50 ", g_type, len(dg))
            dL.append(np.array(dg))

        schema_debug ("Flag 676.60 ")
        dg1, dg2 = dL[0], dL[1]
        
        if d2 is None:
            rp = scipy.stats.pearsonr(dg1, scipy.stats.rankdata(dg1))[0]
            rs = None
        else:
            rp = scipy.stats.pearsonr(dg1, dg2)[0]
            rs = scipy.stats.spearmanr(dg1, dg2)[0]

        schema_debug ("Flag 676.80 ", rp, rs)
        corrL.append( (rp,rs))

    return corrL








