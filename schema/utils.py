#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, copy, multiprocessing
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse
import os.path
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix
from sklearn.preprocessing import normalize
from collections import defaultdict


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

                    print ("Flag 324.112 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                    mdlty["gene_id"] = mdlty["gene_id"].apply(lambda v: v.split('.')[0])
                    mdlty_idx[~(mdlty["gene_id"].isin(dref["ensembl_id"]))] = False

                    print ("Flag 324.113 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                    mdlty["index"] = np.arange(mdlty.shape[0])
                    mdlty = (pd.merge(mdlty, dref, left_on="gene_id", right_on="ensembl_id", how="left"))
                    mdlty = mdlty.drop_duplicates("index").sort_values("index").reset_index(drop=True).drop(columns=["index"])

                    print ("Flag 324.114 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))

                    def f_is_standard_chromosome(v):
                        try:
                            assert v[:3]=="chr" and ("_" not in v) and (v[3] in ['X','Y'] or int(v[3:])<=22 )
                            return True
                        except:
                            return False

                    mdlty_idx[~(mdlty["chr"].apply(f_is_standard_chromosome))] = False

                    print ("Flag 324.115 filtered {0}:{1}:{2} mdlty".format(len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                else:                    
                    mdlty_idx = np.full((mdlty.shape[0],),True)
                    if f_mdlty_filter is not None:
                        mdlty_idx = (mdlty.apply(f_mdlty_filter, axis=1, result_type="reduce")).values

                
                    
                # data = pd.DataFrame(data = scipy.io.mmread(glob.glob("{0}/{1}*_count.txt".format(path, gsm))[0]).T.tocsr().astype(np.float_),
                #                     index = cells['sample'],
                #                     columns = mdlty['mdlty_short_name'])

                data = scipy.io.mmread(glob.glob("{0}/{1}*_{2}_count.txt".format(path, gsm, typestr))[0]).T.tocsr().astype(np.float_)

                print ("Flag 324.12 read {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))


                print ("Flag 324.13 filtered  {0}:{1}:{2} cells, {3}:{4}:{5} mdlty".format( len(cell_idx), np.sum(cell_idx), cell_idx.shape, len(mdlty_idx), np.sum(mdlty_idx), mdlty_idx.shape))
                    
                data = data[cell_idx, :]
                data = data[:, mdlty_idx]
                #data = data[cell_idx,:] # mdlty_idx]
                cells = cells[cell_idx].reset_index(drop=True)
                mdlty = mdlty[mdlty_idx].reset_index(drop=True)

                print ("Flag 324.14 filtered down to {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))
                

                print ("Flag 324.15 filtered down to {0} cells, {1} mdlty, {2} data".format(cells.shape, mdlty.shape, data.shape))
                
                sortidx = np.argsort(cells['sample'].values)
                cells = cells.iloc[sortidx,:].reset_index(drop=True)
                data = data[sortidx, :]

                print ("Flag 324.17 \n {0} \n {1}".format( cells.head(2), data[:2,:2]))
                       
                datasets.append((nm, typestr, data, cells, mdlty))
                
                cell_sets.append(set(cells['sample'].values))
                
            except:
                raise 
                #raise ValueError('{0} could not be read in {1}'.format(nm, path))

        common_cells = list(set.intersection(*cell_sets))
        print ("Flag 324.20 got {0} common cells {1}".format( len(common_cells), common_cells[:10]))

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
            print ("Flag 324.205 got {0} {1} {2} {3}".format( nm, cells.shape, len(cidx), np.sum(cidx)))
                    
            data = data[cidx,:]
            cells = cells.iloc[cidx,:].reset_index(drop=True)
            mdlty = mdlty.set_index('gene_short_name' if typestr=='gene' else 'peak')
            
            if i==0:
                cells = cells.set_index('sample')
                adata = AnnData(X = logcpm(data), obs = cells.copy(deep=True), var = mdlty.copy(deep=True))
                adata.uns["names"] = []

                print ("Flag 324.22 got X {0} obs {1} var {2} uns {3}".format( adata.X.shape, adata.obs.shape, adata.var.shape, list(adata.uns.keys())))
            else:
                for c in cells.columns:
                    if c in adata.obs.columns: continue
                    adata.obs[c] = cells[c]
                adata.uns[nm + ".X"] = logcpm(data)
                adata.uns[nm + ".var"] = mdlty.copy(deep=True)
                adata.uns[nm + ".var.index"] = list(mdlty.index) #scanpy is annoying, it'll convert these to numpy matrices when writing
                adata.uns[nm + ".var.columns"] = list(mdlty.columns)

            adata.obs[typestr + "_log_sumcounts"] = np.log2(np.sum(data,axis=1)+1)
            adata.uns["names"].append(nm)
            adata.uns[nm + ".type"] = typestr
                       
            print ("Flag 324.25 got X {0} obs {1} var {2} uns {3}".format( adata.X.shape, adata.obs.shape, adata.var.shape, list(adata.uns.keys())))
                   
        return adata

    @staticmethod
    def loadAnnData(fpath):
        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 

        adata = sc.read(fpath)
        for k in adata.uns.keys():
            if k.endswith(".var") and isinstance(adata.uns[k], np.ndarray):
                print("Flag 343.100 ", k, adata.uns.keys(), type(adata.uns[k]))
                adata.uns[k] = pd.DataFrame(adata.uns[k], index= adata.uns[k + ".index"], columns = adata.uns[k + ".columns"])
                print("Flag 343.102 ", k, adata.uns.keys(), type(adata.uns[k]))
                for c in  adata.uns[k].columns:
                    if c in ["id","start","end"]:
                        adata.uns[k][c] = adata.uns[k][c].astype(int)
                    if c in ["chr"]:
                        adata.uns[k][c] = adata.uns[k][c].astype(str)
        return adata


    
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
        print ("Flag 321.10012 ", peak)
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
    def computeGeneByFpeakMatrix(adata, peak_func_list, chr_mapping = None, peakList = None, normalize_distwt=True):
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
            
            pXti_ones = np.ones_like(pXti)

            print ("Flag 2.0008 ",i, flush=True)

            for j,f in enumerate(peak_func_list):
                distwt = f(v)
                #print("Flag 2.0010 ", i,j,np.std(distwt), np.mean(distwt)) 
                if np.sum(distwt)>1e-12:
                    if normalize_distwt: distwt = distwt/np.mean(distwt)
                    G = gXt.copy()
                    try:
                        G.data *= distwt.repeat(np.diff(G.indptr))
                    except:
                        G = G*distwt[:,None]
                    # print ("Flag 2.0201 ", len(distwt), np.sum(G.data), G.shape, 
                    #        np.sum(gXt.data),
                    #        G.data.shape, len(np.diff(G.indptr)),
                    #       pXt.shape)
                    gw = G.dot(pXti).ravel()
                    g2p[:,j] += gw/nCells

                    gw_ones = G.dot(pXti_ones).ravel()
                    g2p_wts[:,j] += gw_ones/nCells

                    # print ("Flag 2.0320 ", G.shape, distwt.shape, gXt.shape, pXt.shape, gw.shape, g2p.shape)
                
        return (g2p, g2p_wts)


    
def mprun_geneXfpeak_mtx(adata, n_jobs=8, style=1):
    peak_func_list = [a[0] for a in SciCar.fpeak_list_all]
    print (adata.uns.keys(), adata.uns["atac.var"].head(2))
    chr_mapping = SciCar.getChrMapping(adata)

    assert style in [1,2]

    nPeaks = adata.uns["atac.X"].shape[1]
    l = [list(a) for a in np.array_split(range(nPeaks), 5*n_jobs)]
    #l = [list(a) for a in np.array_split(range(1000), 5*n_jobs)]
    #print("Flag 3343.100 ", l)
    pool =  multiprocessing.Pool(processes = n_jobs)
    lx = pool.map(functools.partial(SciCar.computeGeneByFpeakMatrix, 
                                    adata, peak_func_list, chr_mapping, normalize_distwt= (style==1)), 
                  l)
    # # lx = []
    # # for z in l:
    # #     lx.append(SciCar.computeGeneByFpeakMatrix(adata, peak_func_list, chr_mapping, z))


    if style == 1:
        g2p = None
        for m, _ in lx:
            if g2p is None:
                g2p = m
            else:
                g2p += m

        g2p =  g2p * (1e5/nPeaks)
    elif style==2:
        g2p = None
        g2p_wts = None
        for m, m_wts in lx:
            if g2p is None:
                g2p = m
                g2p_wts = m_wts
            else:
                g2p += m
                g2p_wts += m_wts
        g2p =  (g2p / g2p_wts) * (1e5)
        

    dx = pd.DataFrame(g2p, index=None)
    dx.columns = [a[1] for a in SciCar.fpeak_list_all]
    dx["gene"] = list(adata.var.index)
    dx["ensembl_id"] = list(adata.var.ensembl_id)               
    return dx



def f_helper_mprun_schemawts_1(args):
    ax, dz, dir1, outsfx, min_corr, maxwt, use_first_col = args
    if use_first_col:
        dz_cols = dz.columns[:-2]
        dz_vals = dz.values[:,:-2]
    else:
        dz_cols = dz.columns[1:-2]
        dz_vals = dz.values[:,1:-2]

    import schema_qp
    sqp = schema_qp.SchemaQP(min_corr, maxwt, params= {"dist_npairs": 1000000}, mode="scale")
    dz1 = sqp.fit_transform(dz_vals, [ax], ['feature_vector'], [1])

    wdf = pd.Series(sqp._wts, index=dz_cols).sort_values(ascending=False).reset_index().rename(columns={"index": "fdist",0: "wt"})
    wdf.to_csv("{0}/adata1_sqp_wts_mincorr{1}_maxw{2}_usefirstcol{3}_{4}.csv".format(dir1, min_corr, maxwt, 1 if use_first_col else 0, outsfx), index=False)


def f_helper_mprun_schemawts_2(args):
    ax, dz, dir1, outsfx, min_corr, maxwt, use_first_col, strand, chromosome, gene2fpeak_norm_style = args
    if use_first_col:
        dz_cols = dz.columns[:-2]
        dz_vals = dz.values[:,:-2]
    else:
        dz_cols = dz.columns[1:-2]
        dz_vals = dz.values[:,1:-2]
        
    if int(gene2fpeak_norm_style)==1:
        vstd = np.std(dz_vals.astype(float), axis=0)
        print("Flag 231.020 ", vstd.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vstd)
        
    if int(gene2fpeak_norm_style)==2:
        vstd = np.std(dz_vals.astype(float), axis=0)
        print("Flag 231.022 ", vstd.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vstd)
        vl2norm = np.sqrt(np.sum(np.power(dz_vals.astype(float),2), axis=0))
        print("Flag 231.024 ", vl2norm.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vl2norm)
        
    if int(gene2fpeak_norm_style)==3:
        vl2norm = np.sqrt(np.sum(np.power(dz_vals.astype(float),2), axis=0))
        print("Flag 231.025 ", vl2norm.shape, dz_vals.shape, flush=True)
        dz_vals = dz_vals.copy() / (1e-12 + vl2norm)
        
    import schema_qp
    sqp = schema_qp.SchemaQP(min_corr, maxwt, params= {"dist_npairs": 1000000}, mode="scale")
    dz1 = sqp.fit_transform(dz_vals, [ax], ['feature_vector'], [1])

    print("Flag 231.030 ", min_corr, maxwt, dz_vals.shape, ax.shape, flush=True)
    
    wtsx = np.sqrt(np.maximum(sqp._wts/np.sum(sqp._wts), 0))
    
    wdf = pd.Series(wtsx, index=dz_cols).sort_values(ascending=False).reset_index().rename(columns={"index": "fdist",0: "wt"})
    wdf.to_csv("{0}/adata1_sqp_wts_mincorr{1}_maxw{2}_usefirstcol{3}_strand{4}_chr{5}_norm-gene2fpeak{6}_{7}.csv".format(dir1, min_corr, maxwt, (1 if use_first_col else 0), strand, chromosome,  gene2fpeak_norm_style, outsfx), index=False)


    
def mprun_schemawts_1(adata1, dz, dir1, outsfx, n_jobs=4):
    import schema_qp
    
    pool =  multiprocessing.Pool(processes = n_jobs)
    try:
        ax = np.copy(adata1.X.todense().T)
    except:
        ax = adata1.X.T.copy()

    lx = []
    for mc in [ 0.01, 0.10, 0.20, 0.50, 0.90]:
        for mw in [100, 50, 10]:
            for use_first_col in [True, False]:
                lx.append((ax, dz, dir1, outsfx, mc, mw, use_first_col))
                
    pool.map(f_helper_mprun_schemawts_1, lx)
    

    
def mprun_schemawts_2(adata1, dz, dir1, outsfx, n_jobs=4):
    import schema_qp
    
    pool =  multiprocessing.Pool(processes = n_jobs)
    try:
        ax = np.copy(adata1.X.todense().T)
    except:
        ax = adata1.X.T.copy()

    nGenes = adata1.var.shape[0]
    chrx = adata1.var["chr"].apply(lambda s: s.replace("chr",""))
    
    lx = []
    
            
    for gene2fpeak_norm_style in [0, 1, 2, 3]:  # also tried [0,1]
        dz2 = dz
        # if gene2fpeak_norm_style==True:
        #     dz2  = dz.copy(deep=True)
        #     dz2  = dz2 / (1e-12 + dz2.std())
            
        for strand in ["both","plus", "minus"]:
            gidx_strand = np.full(nGenes, True)
            if strand == "plus":  gidx_strand = np.where(adata1.var["strand"]=="+", True, False)
            if strand == "minus": gidx_strand = np.where(adata1.var["strand"]=="-", True, False)

            for chromosome in ["all", "1--8","9--16","17--23"]:
                if (strand!="both") and chromosome!="all": continue
                
                gidx_chr = np.full(nGenes, True)
                if chromosome=="1--8":   gidx_chr = chrx.isin("1,2,3,4,5,6,7,8".split(","))
                if chromosome=="9--16":  gidx_chr = chrx.isin("9,10,11,12,13,14,15,16".split(","))
                if chromosome=="17--23": gidx_chr = chrx.isin("17,18,19,20,21,22,X,Y".split(","))

                for mc in [ 0.01, 0.25, 0.50, 0.75, 0.875, 0.90]:
                    if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and mc not in [0.01, 0.20]: continue
                    
                    for mw in [100, 50, 30, 20, 10]:
                        #if strand=="both" and chromosome=="all" and gene2fpeak_norm_style==False and mw != 100: continue
                        if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and mw!=100: continue
                    
                        for use_first_col in [True, False]:
                            #if strand=="both" and chromosome=="all" and gene2fpeak_norm_style==False and use_first_col != True: continue
                            if (strand!="both" or chromosome!="all" or gene2fpeak_norm_style!=1) and use_first_col!=True: continue
                            
                            gidx = gidx_strand & gidx_chr
                            print("Flag 3312.040 ", gidx.shape, np.sum(gidx), ax.shape, dz2.shape, flush=True)
                            lx.append((ax[gidx,:], dz2[dz2.ensembl_id.isin(adata1.var["ensembl_id"][gidx])], dir1, outsfx, mc, mw, use_first_col, strand, chromosome, gene2fpeak_norm_style))
                            print("Flag 3312.050 ", np.sum(gidx), lx[-1][2:], flush=True)
                            
    pool.map(f_helper_mprun_schemawts_2, lx)

    
#################################################################################

if __name__ == "__main__":

    mode = sys.argv[1]
    outsfx = sys.argv[2]
    if mode=="1":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 24)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)

    if mode=="2":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())
        dx = mprun_geneXfpeak_mtx(adata1, 24)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_gexp-standardized_{0}.csv".format(outsfx), index=False)

    if mode=="3":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191126_1900.csv".format(dir1))
        mprun_schemawts_1(adata1, dz, dir1, outsfx, 4)

    if mode=="21":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())
        dx = mprun_geneXfpeak_mtx(adata1, 36)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_gexp-standardized_{0}.csv".format(outsfx), index=False)

    if mode=="22":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 36)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)

    if mode=="221":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dx = mprun_geneXfpeak_mtx(adata1, 36, style=2)
        dx.to_csv("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1_gene2fpeak_mtx_{0}.csv".format(outsfx), index=False)


    if mode=="23":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)

    if mode=="231" or mode=="2311":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")

        #adata1.X = sklearn.preprocessing.StandardScaler().fit_transform(adata1.X.todense())

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))
        if mode=="2311":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_gexp-standardized_20191202-1245.csv".format(dir1))

        mprun_schemawts_2(adata1, dz, dir1,  "mode" + mode + "_" + outsfx, 5)

    if mode=="232" or mode=="2321":
        adata1 = SciCar.loadAnnData("/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car/adata1.h5ad")

        import matplotlib
        matplotlib.use("agg") #otherwise scanpy tries to use tkinter which has issues importing
        import scanpy as sc 
        sc.pp.highly_variable_genes(adata1, n_top_genes=5000, inplace=True)

        ax_l2norm = np.sqrt(np.sum(np.power(adata1.X.todense(),2), axis=0)) #np.sqrt(np.sum(adata1.X**2, axis=0))
        print ("Flag 2321.100 ", ax_l2norm.shape, flush=True)
        adata1.X = adata1.X.todense() / (1e-12 + ax_l2norm)

        adata1 = adata1[:, adata1.var["highly_variable"]]
        dir1="/afs/csail.mit.edu/u/r/rsingh/work/schema/data/sci-car"
        dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_20191202-1400.csv".format(dir1))
        if mode=="2321":
            dz = pd.read_csv("{0}/adata1_gene2fpeak_mtx_gexp-standardized_20191202-1245.csv".format(dir1))

        dz = dz[dz.ensembl_id.isin(adata1.var["ensembl_id"])].reset_index(drop=True)
        mprun_schemawts_2(adata1, dz, dir1, "mode" + mode + "_" + outsfx, 5)
