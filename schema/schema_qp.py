#!/usr/bin/env python


###################################################################
## Primary Author:  Rohit Singh rsingh@alum.mit.edu
## Co-Authors: Ashwin Narayan, Brian Hie {ashwinn,brianhie}@mit.edu
## License: MIT
## Repository:  http://github.io/rs239/schema
###################################################################

import pandas as pd
import numpy as np
import scipy, sklearn
import os, sys, string, fileinput, glob, re, math, itertools, functools, copy, logging
import sklearn.decomposition, sklearn.preprocessing, sklearn.linear_model, sklearn.covariance
import cvxopt


schema_loglevel = logging.WARNING  #can be logging.INFO, .DEBUG or .ERROR


def schema_debug(*args, **kwargs):
    if schema_loglevel <= logging.DEBUG: print("DEBUG: ", *args, **kwargs)

def schema_info(*args, **kwargs):
    if schema_loglevel <= logging.INFO: print("INFO: ", *args, **kwargs)

def schema_warning(*args, **kwargs):
    if schema_loglevel <= logging.WARNING: print("WARNING: ", *args, **kwargs)

def schema_error(*args, **kwargs):
    if schema_loglevel <= logging.ERROR: print("ERROR: ", *args, **kwargs)

    
########## for maintenance ###################
# def noop(*args, **kwargs):
#     pass
#
# logging.info = print
# logging.debug = noop
##############################################


class SchemaQP:
    """Schema is a general algorithm for integrating heterogeneous data 
           modalities. It has been specially designed for multi-modal 
           single-cell biological datasets, but should work in other contexts too.
       This version is based on a Quadratic Programming Framework.

       Source code, API documentation and Examples available at: 
       https://github.com/rs239/schema 

       It is described in the paper “Schema: A general framework for integrating 
           heterogeneous single-cell modalities” 
       https://www.biorxiv.org/content/10.1101/834549v1/

       This class provides a sklearn type fit+transform API for affine 
           transformations of input datasets such that the transformed data 
           is in agreement with all the input datasets.


      Example
      --------
           sqp = SchemaQP( min_desired_corr = 0.9) 

           Fit
           ---
           # scobj is a scanpy/anndata obj; 3 secondary datasets each given equal wt
           sqp.fit(scobj.X, 
                   [scobj.obs.col1.values, scobj.obs.col2.values, datasetY.values], 
                   ["categorical", "numeric", "feature_vector", "feature_vector_categorical"]) 

           #or 

           # df is a pd.DataFrame, srs is a pd.Series, -1 means try to disagree
           sqp.fit( df.values, [srs.values], ['numeric'], [-1]) 

           Transform
           ----------
           dnew = sqp.transform(scobj.X) 
        
           Fit+Transform
           -------------
           dnew = sqp.fit_transform( df.values, [srs.values], ['numeric'], [-1]) 
    
#### Parameters

`min_desired_corr`: `float` in [0,1)

    The minimum desired correlation between squared L2 distances in the transformed space
    and distances in the original space.


    RECOMMENDED VALUES: At first, you should try a range of values (e.g., 0.99, 0.90, 0.50).
                        This will give you a sense of what might work well for your data.
                        After this, you can progressively narrow down your range.
                        In typical use-cases of large biological datasets,
                        high values (> 0.80) will probably work best.


`w_max_to_avg`: `float` >1, optional (default: 100)

     Sets the upper-bound on the ratio of w's largest element to w's avg element.
     Making it large will allow for more severe transformations.

    RECOMMENDED VALUES: Start by keeping this constraint very loose; the default value (100) does
                        this, ensuring that min_desired_corr remains the binding constraint.
                        Later, as you get a better sense for the right min_desired_corr values
                        for your data, you can experiment with this too.

                        To really constrain this, set it in the (1-5] range, depending on
                        how many features you have.


`params`: `dict` of key-value pairs, optional (see defaults below)

     Additional configuration parameters.
     Here are the important ones:
       * decomposition_model: "pca" or "nmf" (default=pca)
       * num_top_components: (default=50) number of PCA (or NMF) components to use
           when mode=="affine".

     You can ignore the rest on your first pass; the default values are pretty reasonable:
       * dist_npairs: (default=2000000). How many pt-pairs to use for computing pairwise distances
           value=None means compute exhaustively over all n*(n-1)/2 pt-pairs. Not recommended for n>5000.
           Otherwise, the given number of pt-pairs is sampled randomly. The sampling is done
           in a way in which each point will be represented roughly equally.
       * scale_mode_uses_standard_scaler: 1 or 0 (default=0), apply the standard scaler
           in the scaling mode
       * do_whiten: 1 or 0 (default=1). When mode=="affine", should the change-of-basis loadings
           be made 1-variance?


`mode`: `string` one of {`'affine'`, `'scale'`}, optional (default: `'affine'`)

    Whether to perform a general affine transformation or just a scaling transformation

    * 'scale' does scaling transformations only.
    * 'affine' first does a mapping to PCA or NMF space (you can specify n_components)
         It then does a scaling transform in that space and then maps everything back to the
         regular space, the final space being an affine transformation

    RECOMMENDED VALUES: 'affine' is the default, which uses PCA or NMF to do the change-of-basis.
                        You'll want 'scale' only in one of two cases:
                         1) You have some features on which you directly want Schema to compute
                            feature-weights.
                         2) You want to do a change-of-basis transform other PCA or NMF. If so, you will
                            need to do that yourself and then call SchemaQP with the transformed
                            primary dataset with mode='scale'.

#### Returns

    A SchemaQP object on which you can call fit(...), transform(...) or fit_transform(....).

    """

    
    def __init__(self, min_desired_corr, w_max_to_avg=100, params={}, mode="affine"):
        if not (mode in ['scale', 'affine']): raise ValueError("'mode' must be one of ['affine','scale']")
        if not (w_max_to_avg > 1): raise ValueError("'w_max_to_avg' must be greater than 1")
        if not (0 <= min_desired_corr < 1): raise ValueError("'min_desired_corr' must be in the range [0,1)") 

        self._mode = mode
        self._w_max_to_avg = w_max_to_avg
        print("Flag 456.10  ", self._w_max_to_avg)
        self._min_desired_corr = min_desired_corr
        self._std_scaler = None
        self._decomp_mdl = None


        # defaults
        self._params = {"decomposition_model": "pca",
                        "num_top_components": 50, #default is 20
                        "dist_npairs": 2000000,
                        "d0_dist_transform": None,
                        "secondary_data_dist_transform_list": None,
                        "d0_orig_transformed_corr": None,
                        "dist_df_frac": 1.0,
                        "scale_mode_uses_standard_scaler": 0,
                        "do_whiten": 1,
                        "require_nonzero_lambda": 0,
                        "d0_type_is_feature_vector_categorical": 0,
        }


        self._params.update(params) # overwrite with user settings
        
        if not (self._params["decomposition_model"].lower() in ["pca","nmf"]):
            raise ValueError("For change-of-basis transforms other than PCA/NMF, please do the transformation yourself and use SchemaQP with mode='scale'")
        
        if not (self._params['num_top_components'] is None or self._params["num_top_components"] >= 2): raise ValueError('Need num_top_components >= 2')


        
    
    def fit(self, d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list = None, d0 = None, d0_dist_transform=None, secondary_data_dist_transform_list=None):
        """
Given the primary dataset 'd' and a list of secondary datasets, fit a linear transformation (d*) of
   'd' such that the correlation between squared pairwise distances in d* and those in secondary datasets
    is maximized while the correlation between the primary dataset d and d* remains above
    min_desired_corr


#### Parameters

`d`: A numpy 2-d `array`

    The primary dataset (e.g. scanpy/anndata's .X).
    The rows are observations (e.g., cells) and the cols are variables (e.g., gene expression).
    The default distance measure computed is L2: sum((point1-point2)**2). See d0_dist_transform.


`secondary_data_val_list`: `list` of 1-d or 2-d numpy `array`s, each with same number of rows as `d`

    The secondary datasets you want to align the primary data towards.
    Columns in scanpy's .obs variables work well (just remember to use .values)


`secondary_data_type_list`: `list` of `string`s, each value in {'numeric','feature_vector','categorical', 'feature_vector_categorical'}

    The list's length should match the length of secondary_data_val_list

    * 'numeric' means you're giving one floating-pt value for each obs.
          The default distance measure is L2:  (point1-point2)**2
    * 'feature_vector' means you're giving some multi-dimensional representation for each obs.
          The default distance measure is L2: sum_{i}((point1[i]-point2[i])**2)
    * 'feature_vector_categorical' means you're giving some multi-dimensional representation for each obs.
          Each column can take on categorical values, so the distance between two points is sum_{i}(point1[i]==point2[i])
    * 'categorical' means that you are providing label information that should be compared for equality.
          The default distance measure is: 1*(val1!=val2)


`secondary_data_wt_list`: `list` of `float`s, optional (default: `None`)

    User-specified wts for each dataset. If 'None', the wts are 1.
    If specified, the list's length should match the length of secondary_data_wt_list

    NOTE: you can try to get a mapping that *disagrees* with a dataset_info instead of *agreeing*.
      To do so, pass in a negative number (e.g., -1)  here. This works even if you have just one secondary
      dataset


`d0`: A 1-d or 2-d numpy array, same number of rows as 'd', optional (default: `None`)

    An alternative representation of the primary dataset.

    HANDLE WITH CARE! Most likely, you don't need this parameter.
    This is useful if you want to provide the primary dataset in two forms: one for transforming and
    another one for computing pairwise distances to use in the QP constraint; if so, 'd' is used for the
    former, while 'd0' is used for the latter


`d0_dist_transform`: a function that takes a non-negative float as input and
                    returns a non-negative float, optional (default: `None`)


    HANDLE WITH CARE! Most likely, you don't need this parameter.
    The transformation to apply on d or d0's L2 distances before using them for correlations.


`secondary_data_dist_transform`: `list` of functions, each taking a non-negative float and
                                 returning a non-negative float, optional (default: `None`)

    HANDLE WITH CARE! Most likely, you don't need this parameter.
    The transformations to apply on secondary dataset's L2 distances before using them for correlations.
    If specified, the length of the list should match that of secondary_data_val_list


#### Returns:

    None
         """
        
        if not (d.ndim==2): raise ValueError('d should be a 2-d array')
        if not (len(secondary_data_val_list) >0): raise ValueError('secondary_data_val_list can not be empty')

        if not (len(secondary_data_val_list)==len(secondary_data_type_list)):
            raise ValueError('secondary_data_type_list should have the same length as secondary_data_val_list')
        
        if not (secondary_data_wt_list is None or len(secondary_data_wt_list)==len(secondary_data_val_list)):
            raise ValueError('secondary_data_wt_list should have the same length as secondary_data_val_list')

        for i in range(len(secondary_data_val_list)):
            if not (secondary_data_type_list[i] in ['categorical','numeric','feature_vector']):
                raise ValueError('{0}-th entry in secondary_data_type_list is invalid'.format(i+1))
             
            if not (secondary_data_val_list[i].shape[0] == d.shape[0]):
                raise ValueError('{0}-th entry in secondary_data_val_list has incorrect rows'.format(i+1))
            
            if not ((secondary_data_type_list[i]=='categorical' and secondary_data_val_list[i].ndim==1) or
                    (secondary_data_type_list[i]=='numeric' and secondary_data_val_list[i].ndim==1) or
                    (secondary_data_type_list[i]=='feature_vector' and secondary_data_val_list[i].ndim==2) or
                    (secondary_data_type_list[i]=='feature_vector' and secondary_data_val_list[i].ndim==2)):
                raise ValueError('{0}-th entry in secondary_data_val_list does not match specified type'.format(i+1))

            
        if not (d0 is None or d0.shape[0] == d.shape[0]): raise ValueError('d0 has incorrect rows')

            

        self._params["d0_dist_transform"] = d0_dist_transform
        self._params["secondary_data_dist_transform_list"] = secondary_data_dist_transform_list
        
        if self._mode=="scale":
            self._fit_scale(d, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)
        else: #affine
            self._fit_affine(d, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)



            
            
    def transform(self, d):
        """
Given a dataset `d`, apply the fitted transform to it


#### Parameters

`d`:  a numpy 2-d array with same number of columns as primary dataset `d` in the fit(...)

    The rows are observations (e.g., cells) and the cols are variables (e.g., gene expression).


#### Returns

 a 2-d numpy array with the same shape as `d`
         """
        
        if self._mode=="scale":
            if not (d.shape[1] == len(self._wts)): raise ValueError('Number of columns in d is incorrect')
            dx = self._std_scaler.transform(d)
            return np.multiply(dx, np.sqrt(self._wts))
        
        else: #affine
            if not (d.shape[1] == self._decomp_mdl.components_.shape[1]): raise ValueError('Number of columns in d is incorrect')
            dx = self._decomp_mdl.transform(d)
            return np.multiply(dx, np.sqrt(self._wts))

        

        
    def fit_transform(self, d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list = None, d0 = None, d0_dist_transform=None, secondary_data_dist_transform_list=None):
        """
        Calls fit(..) with exactly the arguments given; then calls transform(d).
        See documentation for fit(....) and transform(...) respectively.

        """

        self.fit(d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list, d0, d0_dist_transform, secondary_data_dist_transform_list)
        return self.transform(d)


    ###################################################################
    ######## "private" methods below. Not that Python cares... ########
    ###################################################################
    def _getDistances(self, D, d0_in, G, nPointPairs):
        """
        Compute the various distances between point pairs in D
        D is a NxK numpy type 2-d array, with N points, each of K dimensions
        d0_in could be None, in which D is used to compute distances for the original space.
           Otherwise, d0_in is Nxr (r>=1) array and original-space distances are computed from this. 
        G is list, with each entry a 3-tuple: (g_val, g_type, gamma_g)
           each 3-tuple corresponds to one dimension of side-information
             g_val is Nx1 numpy type 1-d vector of values for the N points, in the same order as D
             g_type is "numeric", "categorical", or "feature_vector"
             gamma_g is the relative wt you want to give this column. You can leave 
               it as None for all 3-tuples, but not for just some. If you leave them
               all as None, the system will set wts to 1  
        nPointPairs is the number of point pairs you want to evaluate over. If it is None,
             the system will generate over all possible pairs
        """
        ##############################
        # symbology
        # uv_dist = nPointPairs x K matrix of squared distances between point pairs, along K dimensions
        # uv_dist_mean = 1 x K vector: avg of uv_dist along each dimension 
        # uv_dist_centered = nPointPairs x K matrix with uv_dist_mean subtracted from each row
        # d0 = nPointPairs x 1 vector: squared distances in the current space
        # z_d0 = z-scored version of  d0
        # dg = nPointPairs x 1 vector: squared distances (or class-match scores) for grouping g
        # z_dg = z-scored version of dg
        ##############################
        
        schema_info ("Flag 232.12 ", len(G), G[0][0].shape, G[0][1], G[0][2])
        gamma_gs_are_None = True
        for i,g in enumerate(G):
            g_val, g_type, gamma_g = g
                
            assert g_type in ['categorical','numeric','feature_vector']
            
            if i==0 and gamma_g is not None:
                gamma_gs_are_None = False
            if gamma_g is None: assert gamma_gs_are_None
            if gamma_g is not None: assert (not gamma_gs_are_None)
            

        N, K = D.shape[0], D.shape[1]

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
            
        # NxK matrix of square distances along dimensions
        if int(self._params.get("d0_type_is_feature_vector_categorical",1)) > 0.5:
            uv_dist = (D[i_u,:] == D[i_v,:]).astype(np.float64)
        else:
            uv_dist = (D[i_u,:].astype(np.float64) - D[i_v,:].astype(np.float64))**2 


        # scale things so QP gets ~1-3 digit numbers, very large or very small numbers cause numerical issues         
        uv_dist = uv_dist/ (np.sqrt(uv_dist.shape[0])*uv_dist.ravel().mean())  
        
        # center uv_dist along each dimension
        uv_dist_mean = uv_dist.mean(axis=0) #Kx1 vector, this is the mean of (u_i - v_i)^2, not sqrt((u_i - v_i)^2) 
        uv_dist_centered = uv_dist - uv_dist_mean
        
        # square distances in the current metric space
        if d0_in is None:
            d0 = uv_dist.sum(axis=1) # Nx1
        else:
            dx0 = (d0_in[i_u] - d0_in[i_v])**2
            if dx0.ndim==2:
                d0 = dx0.sum(axis=1)
            else:
                d0 = dx0

        d0_orig = d0.copy()            
        f_dist_d0 = self._params["d0_dist_transform"]
        self._params["d0_orig_transformed_corr"] = 1.0

        if f_dist_d0 is not None:
            d0 = f_dist_d0(d0)
            cr = (np.corrcoef(d0_orig, d0))[0,1]
            if cr<0:
                # it's an error if you specify a d0 such that corr( pairwisedist(d), pairwise(d0)) is negative to begin with
                schema_error("d0_dist_transform inverts the correlation structure {0}.".format(cr))
                raise ValueError("""d0_dist_transform inverts the correlation structure. Aborting...
                                    It's an error if you specify a d0 such that corr( pairwisedist(d), pairwise(d0)) is negative to begin with""")
            self._params["d0_orig_transformed_corr"] = cr

            
        z_d0  = (d0 - d0.mean())/d0.std()

        schema_info("Flag 2090.20 Initial corr to d0", np.corrcoef(uv_dist_centered.sum(axis=1), z_d0))
        
        
        l_z_dg = []
        for ii, gx in enumerate(G):
            g_val, g_type, gamma_g = gx
            
            if self._params["secondary_data_dist_transform_list"] is not None:
                f_dist_dg = self._params["secondary_data_dist_transform_list"][ii]
            else:
                f_dist_dg = None

            schema_info ("Flag 201.80 ", g_val.shape, g_type, gamma_g, f_dist_dg)
            
            if g_type == "categorical":
                dg = 1.0*( g_val[i_u] != g_val[i_v]) #1.0*( g_val[i_u].toarray() != g_val[i_v].toarray())
            elif g_type == "feature_vector":
                print (g_val[i_u].shape, g_val[i_v].shape)
                dg = np.ravel(np.sum(np.power(g_val[i_u].astype(np.float64) - g_val[i_v].astype(np.float64),2), axis=1))
            elif g_type == "feature_vector_categorical":
                print (g_val[i_u].shape, g_val[i_v].shape)
                dg = np.ravel(np.sum((g_val[i_u] == g_val[i_v]).astype(np.float64), axis=1))
            else:  #numeric
                dg = (g_val[i_u].astype(np.float64) - g_val[i_v].astype(np.float64))**2   #(g_val[i_u].toarray() - g_val[i_v].toarray())**2            

            if f_dist_dg is not None:
                dg = f_dist_dg(dg)
                
            z_dg = (dg - dg.mean())/dg.std()
            if gamma_g is not None:
                z_dg *= gamma_g
                
            l_z_dg.append(z_dg)

        schema_info ("Flag 201.99 ", uv_dist_centered.shape, z_d0.shape, len(l_z_dg), l_z_dg[0].shape)
        schema_info ("Flag 201.991 ", uv_dist_centered.mean(axis=0))
        schema_info ("Flag 201.992 ", uv_dist_centered.std(axis=0))
        schema_info ("Flag 201.993 ", z_d0[:10], z_d0.mean(), z_d0.std())
        schema_info ("Flag 201.994 ", l_z_dg[0][:10], l_z_dg[0].mean(), l_z_dg[0].std())
        return (uv_dist_centered, z_d0, l_z_dg)


    
    
    def _prepareQPterms(self, uv_dist_centered, z_d0, l_z_dg):
        nPointPairs = uv_dist_centered.shape[0]
        
        l_g = []
        for i,z_dg in enumerate(l_z_dg):
            l_g.append( np.sum(uv_dist_centered * z_dg[:,None], axis=0) )
            
        q1 = l_g[0]
        for v in l_g[1:]:
            q1 += v
        
        P1 = np.matmul( uv_dist_centered.T,  uv_dist_centered)

        g1 = np.sum(uv_dist_centered * z_d0[:,None], axis=0)
        h1 = np.sum(g1)

        return (P1, q1, g1, h1, nPointPairs)

    

    def _computeSolutionFeatures(self, w, P1, q1, g1, nPointPairs):
        K = len(w)
        #print ("Flag 569.20 ", w.shape, P1.shape, q1.shape, g1.shape, np.reshape(w,(1,K)).shape)
        
        newmetric_sd = np.sqrt( np.matmul(np.matmul(np.reshape(w,(1,K)), P1), np.reshape(w,(K,1)))[0] / nPointPairs)
        #oldnew_corr = (np.dot(w,g1)/nPointPairs)/newmetric_sd
        oldnew_corr = ((np.dot(w,g1)/nPointPairs)/newmetric_sd) / (self._params["d0_orig_transformed_corr"])

        groupcorr_score = (np.dot(w,q1)/nPointPairs)/newmetric_sd
        return {"w": w, "distcorr": oldnew_corr[0], "objval": groupcorr_score[0]}
        

    
    def _doQPSolve(self, P1, q1, g1, h1, nPointPairs, lambda1, alpha, beta):
        #https://cvxopt.org/examples/tutorial/qp.html and https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
        #  the cvx example switches P & q to Q & p
        
        from cvxopt import matrix, solvers
        solvers.options["show_progress"] = False


        K = len(g1)

        I_K = np.diagflat(np.ones(K).astype(np.float64))
        #P = 2* (lambda1*matrix(I_K) + beta*matrix(P1))
        P = matrix(2* ((I_K*lambda1) + (P1*beta)))
        
        q = -matrix(q1 + 2*lambda1*np.ones(K))
        
        G0 = np.zeros((K+1,K)).astype(np.float64)
        for i in range(K):
            G0[i,i] = 1.0
        G0[-1,:] = g1

                
        G = -matrix(G0) #first K rows say -w_i <= 0. The K+1'th row says -corr(newdist,olddist) <= -const
        h = matrix(np.zeros(K+1))
        h[-1] = -alpha*h1
        
        
        schema_debug("Flag 543.70 ", P1.shape, q1.shape, g1.shape, h1, nPointPairs, lambda1 , alpha, beta, P.size, q.size, G0.shape, G.size, h.size) #K, P.size, q.size, G.size, h.size)
        sol=solvers.qp(P, q, G, h)
        solvers.options["show_progress"] = True

        w = np.array(sol['x']).ravel()
        s = self._computeSolutionFeatures(w, P1, q1, g1, nPointPairs)
        return s

    
    def _summarizeSoln(self, soln, free_params):
        retl = []
        retl.append(("w_max_to_avg", (max(soln["w"])/np.nanmean(soln["w"]))))
        retl.append(("w_num_zero_dims", sum(soln["w"] < 1e-5)))
        retl.append(("d0_orig_transformed_corr", self._params["d0_orig_transformed_corr"]))
        retl.append(("distcorr", soln["distcorr"]))
        retl.append(("objval", soln["objval"]))
        retl.append(("lambda", free_params["lambda"]))
        retl.append(("alpha", free_params["alpha"]))
        retl.append(("beta", free_params["beta"]))
        
        return retl
                    

    
    def _doQPiterations(self, P1, q1, g1, h1, nPointPairs, max_w_wt, min_desired_oldnew_corr):
        solutionList = []

        schema_info('Progress bar (each dot is 10%): ', end='', flush=True)

        alpha = 1.0  #start from one (i.e. no limit on numerator and make it bigger)
        while alpha > 1e-5:
            soln, param_settings = self._iterateQPLevel1(P1, q1, g1, h1, nPointPairs, max_w_wt, min_desired_oldnew_corr, alpha)
            
            if soln is not None and soln["distcorr"] >= min_desired_oldnew_corr:
                solutionList.append((-soln["objval"], soln, param_settings))

            alpha -= 0.1
            schema_info('.', end='', flush=True)
        try:
            solutionList.sort(key=lambda v: v[0]) #find the highest score                

            schema_info(' Done\n', end='', flush=True)
            return (solutionList[0][1], solutionList[0][2])
        except:
            schema_info(' Done\n', end='', flush=True)
            #raise
            return (None, {})

        

        
    def _iterateQPLevel1(self, P1, q1, g1, h1, nPointPairs, max_w_wt, min_desired_oldnew_corr, alpha):
        solutionList = []

        beta = 1e6  #start from a large value and go towards zero (a large value means the denominator will need to be small)
        while beta > 1e-6:
            try:
                soln, param_settings = self._iterateQPLevel2(P1, q1, g1, h1, nPointPairs, max_w_wt, alpha, beta)
            except Exception as e:
                schema_warning ("Flag 110.50 crashed in _iterateQPLevel2. Trying to continue...", P1.size, q1.size, g1.size, max_w_wt, alpha, beta)
                schema_info(e)
                beta *= 0.5
                continue
            
            if soln["distcorr"] >= min_desired_oldnew_corr:
                solutionList.append((-soln["objval"], soln, param_settings))

            beta *= 0.5
        
        try:
            solutionList.sort(key=lambda v: v[0]) #find the highest score

            schema_info("Flag 110.60 beta: ", "NONE" if not solutionList else self._summarizeSoln(solutionList[0][1], solutionList[0][2]))
            return (solutionList[0][1], solutionList[0][2])
        except:
            #raise
            return (None, {})



    def _iterateQPLevel2(self, P1, q1, g1, h1, nPointPairs, max_w_wt, alpha, beta):
        lo=1e-10 #1e-6
        solLo = self._doQPSolve(P1, q1, g1, h1, nPointPairs, 1/lo, alpha, beta)
        scalerangeLo = (max(solLo["w"])/np.nanmean(solLo["w"]))
        
        hi=1e9 #1e6
        solHi = self._doQPSolve(P1, q1, g1, h1, nPointPairs, 1/hi, alpha, beta)
        scalerangeHi = (max(solHi["w"])/np.nanmean(solHi["w"]))

        if scalerangeHi < max_w_wt:
            solZero = self._doQPSolve(P1, q1, g1, h1, nPointPairs, 0, alpha, beta)
            scalerangeZero = (max(solZero["w"])/np.nanmean(solZero["w"]))
            if scalerangeZero < max_w_wt and int(self._params.get("require_nonzero_lambda",0))==0:
                return solZero, {"lambda": 0, "alpha": alpha, "beta": beta} 
            else:
                return solHi, {"lambda": 1/hi, "alpha": alpha, "beta": beta}
            
        if scalerangeLo > max_w_wt: return solLo, {"lambda": 1/lo, "alpha": alpha, "beta": beta}
        
        niter=0
        while (niter < 60 and (hi/lo -1)>0.001):
            mid = np.exp((np.log(lo)+np.log(hi))/2)
            solMid = self._doQPSolve(P1, q1, g1, h1, nPointPairs, 1/mid, alpha, beta)
            scalerangeMid = (max(solMid["w"])/np.nanmean(solMid["w"]))
            
            if (scalerangeLo <= max_w_wt <= scalerangeMid):
                hi = mid
                scalerangeHi = scalerangeMid
                solHi = solMid
            else:
                lo = mid
                scalerangeLo = scalerangeMid
                solLo = solMid

            niter += 1
            schema_debug ("Flag 42.113 ", niter, lo, mid, hi, max(solLo["w"]), min(solLo["w"]), max(solHi["w"]), min(solHi["w"]), scalerangeMid)
        return solLo, {"lambda": 1/lo, "alpha": alpha, "beta": beta}        

    

    def _fit_helper(self, dx, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list):
        nPointPairs = self._params.get("dist_npairs", 2000000)
        sample_rate = self._params.get("dist_df_frac", 1.0)

        w_max_to_avg= self._w_max_to_avg
        min_desired_corr = self._min_desired_corr

        N = dx.shape[0]

        schema_info ("Flag 102.30 ", dx.shape, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)
        
        G = []
        for i in range(len(secondary_data_val_list)):
            G.append((secondary_data_val_list[i].copy(), secondary_data_type_list[i], None if secondary_data_wt_list is None else secondary_data_wt_list[i]))

        if sample_rate < 0.9999999: 
            idx = np.random.choice(N, size=int(sample_rate*N), replace=False)
            dx = dx[idx,:]
            if d0 is not None:
                d0 = d0[idx,:]
            for i, gx in enumerate(G):
                G[i] = (gx[0][idx], gx[1], gx[2])


        schema_info ("Flag 102.35 ", dx.shape, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)
        uv_dist_centered, z_d0, l_z_dg = self._getDistances(dx, d0, G, nPointPairs)
        schema_info ("Flag 102.36 ", uv_dist_centered.shape, z_d0.shape, len(l_z_dg))
        P1, q1, g1, h1, nPointPairs1 = self._prepareQPterms(uv_dist_centered, z_d0, l_z_dg)
        schema_info ("Flag 102.37 ")
        
        soln, free_params = self._doQPiterations(P1, q1, g1, h1, nPointPairs1, w_max_to_avg, min_desired_corr)
        
        if soln is None:
            raise Exception("Couldn't find valid solution to QP")
        
        schema_info ("Final solution: ", self._summarizeSoln(soln, free_params))
        return soln["w"].ravel(), self._summarizeSoln(soln, free_params)

    
        
    def _fit_scale(self, d, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list):

        dx1 = d.copy()
        if self._params.get("scale_mode_uses_standard_scaler",0)>0:
            self._std_scaler = sklearn.preprocessing.StandardScaler() 
            dx = self._std_scaler.fit_transform(dx1)
        else:
            self._std_scaler = sklearn.preprocessing.StandardScaler(with_mean=False, with_std=False) 
            dx = self._std_scaler.fit_transform(dx1) #this is a little wasteful no-op but makes a code a little easier to manage
            
        print('Running quadratic program...', end='', flush=True)
        s, sl = self._fit_helper(dx, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)
        
        self._wts = np.maximum(s,0)
        self._soln_info = dict(sl)
        print(' done.\n', end='', flush=True)
                
        
    def _fit_affine(self, d, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list):
        do_whiten = self._params.get("do_whiten",1)>0
        ncomp = self._params.get("num_top_components",None) #default is all

        dx1 = d.copy()
        model_type = self._params.get("decomposition_model","pca").lower()

        print('Running change-of-basis transform ({0}, {1} components)...'.format(model_type, ncomp), end='', flush=True)
        
        if model_type=="pca":
            self._decomp_mdl = sklearn.decomposition.PCA(n_components=ncomp, whiten=do_whiten)
            dx = self._decomp_mdl.fit_transform(dx1)
            
        elif model_type=="nmf":
            self._decomp_mdl = sklearn.decomposition.NMF(n_components=ncomp, init=None)
            W = self._decomp_mdl.fit_transform(dx1, W=None, H=None)
            if do_whiten:
                H = self._decomp_mdl.components_
                wsd = W.std(axis=0)
                W = W/wsd
                self._decomp_mdl.components_ *= wsd[:,None]
            dx = W
            
        print(' done.\nRunning quadratic program...', end='', flush=True)
        s, sl = self._fit_helper(dx, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)

        self._wts = np.maximum(s,0)
        self._soln_info = dict(sl)
        print(' done.\n', end='', flush=True)


