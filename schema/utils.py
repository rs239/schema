#!/usr/bin/env python
        
import pandas as pd
import numpy as np
import scipy, sklearn, os, sys, string, fileinput, glob, re, math, itertools, functools, copy
import scipy.stats, sklearn.decomposition, sklearn.preprocessing, sklearn.covariance
from scipy.stats import describe
from scipy import sparse

def sparse_read_csv(filename, index_col=None, verbose=False):
    """
    csr read of file. Does a 100-row load first to get data-types.
    returns sparse matrix, rows and cols.
    based on https://github.com/berenslab/rna-seq-tsne
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



class ScatterPlotAligner:
    """When seeded with a Mx2 matrix (that will be used for a scatter plot), 
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
