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
import os, sys, string, fileinput, glob, re, math, itertools, functools, copy, logging, warnings
import sklearn.decomposition, sklearn.preprocessing
import cvxopt

#### local directory imports ####
oldpath = copy.copy(sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_base_config import *

sys.path = copy.copy(oldpath)
####



class SchemaQP:
    """Schema is a tool for integrating simultaneously-assayed data modalities

    The SchemaQP class provides a sklearn type fit+transform API for constrained
    affine transformations of input datasets such that the transformed data is
    in agreement with all the input datasets.
"""

    
    def __init__(self, min_desired_corr=0.99, mode="affine", params={}):
        """

        :param min_desired_corr: 
            This parameter controls the severity of the primary modality's
            transformation, specifying the minimum required correlation
            between distances in the original space and those in the
            transformed space. It thus controls the trade-off between
            deviating further away from the primary modality's original
            representation and achieving greater agreement with the
            secondary modalities. Values close to one result in lower
            distortion of the primary modality while those close to zero
            enable transformations offering greater agreement
            between the modalities.

            RECOMMENDED VALUES: In typical single-cell use cases, high
            values (> 0.80) will probably work best. With these, the
            distortion will be low, but still be enough for Schema to
            extract relevant information from the secondary modalities.
            Furthermore, the feature weights computed by Schema should
            still be quite infromative.

            The default value of 0.99 is a safe choice to start with; it
            poses low risk of deviating too far from the primary modality.

            Later, you can experiment with a range of values (e.g., 0.95
            0.90, 0.80), or use feature-weights aggregated across an
            ensemble of choices. Alternatively, you can use
            cross-validation to identify the best setting

        :type min_desired_corr: float in [0,1)

        :param mode: 
            Whether to perform a general affine transformation or just a
            scaling transformation

            * `affine` first does a mapping to PCA or NMF space (you can
              specify num_top_components via the `params` argument). Schema does
              a scaling transform in the mapped space and then converts everything
              back to the regular space. The final result is thus an affine
              transformation in the regular space.

            * `scale` does not do a PCA or NMF mapping, and directly applies
              the scaling transformation. **Note**: This can be slow if
              the primary modality's dimensionality is over 100.


            RECOMMENDED VALUES: `affine` is the default. You may need `scale` 
            only in certain cases:

            * You have a limited number of features on which you directly
              want Schema to compute feature-weights.

            * You want to do a change-of-basis transform other PCA or NMF.
              If so, you will need to do that yourself and then call
              SchemaQP with the transformed primary dataset with
              mode='scale'.


        :type mode: string

        :param params: 
             Dictionary of key-value pairs, specifying additional
             configuration parameters. Here are the important ones:

               * `decomposition_model`: "pca" or "nmf" (default=pca)

               * `num_top_components`: (default=50) number of PCA (or NMF)
                 components to use when mode=="affine". We recommend this
                 setting be <= 100. Schema's runtime is quadratic in this
                 number.

             You can ignore the rest on your first pass; the default
             values are pretty reasonable:

               * `dist_npairs`: (default=2000000). How many pt-pairs to use
                 for computing pairwise distances. value=None means
                 compute exhaustively over all n*(n-1)/2 pt-pairs. Not
                 recommended for n>5000.  Otherwise, the given number of
                 pt-pairs is sampled randomly. The sampling is done in a
                 way in which each point will be represented roughly
                 equally.

               * `scale_mode_uses_standard_scaler`: 1 or 0 (default=0),
                 apply the standard scaler in the scaling mode

               * `do_whiten`: 1 or 0 (default=1). When mode=="affine",
                 should the change-of-basis loadings be made 1-variance?

        :type params: dict

        :returns: A SchemaQP object on which you can call fit(...), transform(...) or fit_transform(....).
"""
        if not (mode in ['scale', 'affine']): raise ValueError("'mode' must be one of ['affine','scale']")
        if not (0 <= min_desired_corr < 1): raise ValueError("'min_desired_corr' must be in the range [0,1)") 

        self._mode = mode
        self._w_max_to_avg = 1000 
        schema_info ("Flag 456.10  ", self._w_max_to_avg)
        self._min_desired_corr = min_desired_corr
        self._std_scaler = None
        self._decomp_mdl = None
        self._orig_decomp_mdl = None

        # defaults
        self._params = {"decomposition_model": "nmf",
                        "num_top_components": 20, 
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



    def reset_mincorr_param(self, min_desired_corr):
        """Reset the min_desired_corr. 

        Useful when you want to iterate over multiple choices of this parameter
        but want to re-use the computed PCA or NMF change-of-basis transform.

        :param min_desired_corr: 
            The new value of minimum required correlation between original and transformed distances

        :type min_desired_corr: float in [0,1)

"""
        if min_desired_corr is None or not(0 <= min_desired_corr < 1): raise ValueError("'min_desired_corr' must be between 0 and 1")
        self._min_desired_corr = min_desired_corr



    def reset_maxwt_param(self, w_max_to_avg):
        """ Reset the w_max_to_avg param

        :type w_max_to_avg: float

        :param w_max_to_avg: 
            The upper-bound on the ratio of Schema weights (w's) largest
            element to w's avg element.  Making it large will allow This
            parameter controls the 'deviation' in feature weights and make it
            large will allow for more severe transformations.

            **Handle with care:** We recommend keeping this parameter at its
            default value (1000); that keeps this constraint very loose and
            ensures that min_desired_corr remains the binding constraint.
            Later, as you get a better sense for the right min_desired_corr
            values for your data, you can experiment with this too.  To really
            constrain this, set it in the (1-5] range, depending on how many
            features you have.

"""
        if w_max_to_avg is None or w_max_to_avg <= 1: raise ValueError("'w_max_to_avg' must be either None or greater than 1")
        self._w_max_to_avg = w_max_to_avg
        
    
        
    
    def fit(self, d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list = None, secondary_data_dist_kernels=None, d0 = None, d0_dist_transform=None):
        """Compute the optimal Schema transformation, first performing a
        change-of-basis transformation if required.

        Given the primary dataset `d` and a list of secondary datasets, fit a
        linear transformation (`d_new`) such that the correlation between
        squared pairwise distances in `d_new` and those in secondary datasets is
        maximized while the correlation between the original `d` and the transformed `d_new`
        remains above min_desired_corr.

        The first three arguments are required, the next is useful, and
        the rest should be rarely used.

        :type d: Numpy 2-d `array` or Pandas `dataframe`

        :param d: 
            The primary dataset (e.g. scanpy/anndata's .X).

            The rows are observations (e.g., cells) and the cols are variables (e.g.,
            gene expression).  The default distance measure computed is L2:
            sum((point1-point2)**2). Also see `d0_dist_transform`.


        :type secondary_data_val_list: list of 1-d or 2-d Numpy arrays or Pandas series, each with same number of rows as `d`

        :param secondary_data_val_list: 
            The secondary datasets you want to align the primary data towards.  

            Columns in Anndata .obs or .obsm variables work well.


        :type secondary_data_type_list: list of strings

        :param secondary_data_type_list: 
            The datatypes of the secondary modalities.

            Each element of the list can be one of `numeric,
            feature_vector, categorical, feature_vector_categorical`.  
            The list's length should match the length
            of secondary_data_val_list


            * `numeric`: one floating-pt value for each observation.  The
              default distance measure is Euclidean: (point1-point2)^2
 
            * `feature_vector`: a k-dimensional vector for each
              observation.  The default distance measure is Euclidean:
              sum_{i}((point1[i]-point2[i])^2)

            * `categorical`: a label for each observation.  The default
              distance measure checks for equality: 1*(val1!=val2)

            * `feature_vector_categorical`: a vector of labels for each
              observation.  Each column can take on categorical values, so
              the distance between two points is
              sum_{i}(point1[i]==point2[i])


        :type secondary_data_wt_list: list of floats, **optional**

        :param secondary_data_wt_list: 
            User-specified wts for each secondary dataset (default= list of 1's)

            If specified, the list's length should match the length of
            secondary_data_val_list. When multiple secondary modalities are
            specified, this parameter allows you to control their relative
            weight in seeking an agreement with the primary.

            **Note**: you can try to get a mapping that *disagrees* with a
            dataset_info instead of *agreeing*.  To do so, pass in a
            negative number (e.g., -1) here. This works even if you have
            just one secondary dataset

        :type secondary_data_dist_kernels: list of functions, **optional**

        :param secondary_data_dist_kernels: 
            The transformations to apply on
            secondary dataset's L2 distances before using them for correlations.

            If specified, the length of the list should match that of
            `secondary_data_val_list`. 
            Each function should take a non-negative float and return a non-negative float. 

            **Handle with care:** Most likely, you don't need this parameter.

        :type d0: A 1-d or 2-d Numpy array, **optional** 

        :param d0: 
            An alternative representation of the primary dataset.

            This is useful if you want to provide the primary dataset in two
            forms: one for transforming and another one for computing pairwise
            distances to use in the QP constraint; if so, `d` is used for the
            former, while `d0` is used for the latter.
            If specified, it should have the same number of rows as `d`.

            **Handle with care:** Most likely, you don't need this parameter.


        :type d0_dist_transform: float -> float function, **optional**

        :param d0_dist_transform: 
            The transformation to apply on d or d0's L2
            distances before using them for correlations.

            This function should take a non-negative float as input and return a non-negative float.

            **Handle with care:** Most likely, you don't need this parameter.

        :returns: None
         """
        
        if not (d.ndim==2): raise ValueError('d should be a 2-d array')
        
        if not (len(secondary_data_val_list) >0): raise ValueError('secondary_data_val_list can not be empty')
            
        if not (len(secondary_data_val_list)==len(secondary_data_type_list)):
            raise ValueError('secondary_data_type_list should have the same length as secondary_data_val_list')
        
        if not (secondary_data_wt_list is None or len(secondary_data_wt_list)==len(secondary_data_val_list)):
            raise ValueError('secondary_data_wt_list should have the same length as secondary_data_val_list')

        for i in range(len(secondary_data_val_list)):
            if not (secondary_data_type_list[i] in ['categorical','numeric','feature_vector', 'feature_vector_categorical']):
                raise ValueError('{0}-th entry in secondary_data_type_list is invalid'.format(i+1))
             
            if not (secondary_data_val_list[i].shape[0] == d.shape[0]):
                raise ValueError('{0}-th entry in secondary_data_val_list has incorrect rows'.format(i+1))
            
            if not ((secondary_data_type_list[i]=='categorical' and secondary_data_val_list[i].ndim==1) or
                    (secondary_data_type_list[i]=='numeric' and secondary_data_val_list[i].ndim==1) or
                    (secondary_data_type_list[i]=='feature_vector' and secondary_data_val_list[i].ndim==2) or
                    (secondary_data_type_list[i]=='feature_vector_categorical' and secondary_data_val_list[i].ndim==2)):
                raise ValueError('{0}-th entry in secondary_data_val_list does not match specified type'.format(i+1))

            
        if not (d0 is None or d0.shape[0] == d.shape[0]): raise ValueError('d0 has incorrect rows')

        self._params["d0_dist_transform"] = d0_dist_transform
        self._params["secondary_data_dist_transform_list"] = secondary_data_dist_kernels


        ## Before calling the worker functions, convert to np arrays and check for NaNs
        def fconv_to_np(x):
            if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series): return x.values
            if isinstance(x, list): return np.array(x)
            return x

        def any_nans(npobj):
            if npobj.dtype.kind != 'f': return False
            v = np.sum(npobj)
            return np.isnan(v) or np.isinf(v)
        
        t_d = fconv_to_np(d)
        t_secondary_data_val_list = [ fconv_to_np(v) for v in secondary_data_val_list]
        t_d0 = fconv_to_np(d0)

        if any_nans(t_d): raise ValueError('d should not have any NaN or inf values')
        if t_d0 is not None and any_nans(t_d0): raise ValueError('d0 should not have any NaN or inf values')
        for i, secd in enumerate(t_secondary_data_val_list):
            if any_nans(secd): raise ValueError('{}-th entry in secondary_data_val_list has NaN or inf values'.format(i+1))
                
        
        if self._mode=="scale":
            self._fit_scale(t_d, t_d0, t_secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)
        else: #affine
            self._fit_affine(t_d, t_d0, t_secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)



            
            
    def transform(self, d):
        """Given a dataset `d`, apply the fitted transform to it

        :type d: Numpy 2-d array

        :param d: 
            The primary modality data on which to apply the transformation. 

            `d` must have with same number of columns as in `fit(...)`.
            The rows are observations (e.g., cells) and the cols are variables (e.g., gene expression).


        :returns: a 2-d Numpy array with the same shape as `d`
 """
        
        if self._mode=="scale":
            if not (d.shape[1] == len(self._wts)): raise ValueError('Number of columns in d is incorrect')
            dx = self._std_scaler.transform(d)
            return np.multiply(dx, np.sqrt(self._wts))
        
        else: #affine
            if not (d.shape[1] == self._decomp_mdl.components_.shape[1]): raise ValueError('Number of columns in d is incorrect')
            dx = self._decomp_mdl.transform(d)
            return np.multiply(dx, np.sqrt(self._wts))

        

        
    def fit_transform(self, d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list = None, secondary_data_dist_kernels = None, d0 = None, d0_dist_transform=None):
        """
        Calls fit(..) with exactly the arguments given; then calls transform(d).
        See documentation for fit(....) and transform(...) respectively.

        """

        self.fit(d, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list, secondary_data_dist_kernels, d0, d0_dist_transform)
        return self.transform(d)



    def feature_weights(self, affine_map_style='top-k-loading', k=1):
        """
        Return the feature weights computed by Schema

        If SchemaQP was initialized with `mode=scale`, the weights
        returned are directly the weights from the quadratic programming
        (QP), with a weight > 1 indicating the feature was up-weighted.
        The `affine_map_style` argument is ignored.

        However, if `mode=affine` was used, the QP-computed weights
        correspond to columns of the PCA or NMF decomposition. In that
        case, this functions maps them back to the primary dataset's
        features. This can be done in three different ways, as specified
        by the `affine_map_style` parameter.

        You can build your own mapping from PCA/NMF weights to primary-modality feature
        weights. The instance's `_wts` member is the numpy array that contains
        QP-computed weights, and `_decomp_mdl` is the
        sklearn-computed NMF/PCA decomposition. You can also look at the source code
        of this function to get a sense of how to use them.

        :type affine_map_style: string, one of 'softmax-avg' or 'top-k-rank' or 'top-k-loading', default='top-k-loading'

        :param affine_map_style: 
            Governs how QP-computed weights for PCA/NMF columns are mapped
            back to primary-modality features (typically, genes from a scRNA-seq dataset).

            Default is 'top-k-loading', which considers only the
            top-k PCA/NMF columns by QP-computed weight and computes the
            average loading of a gene across these. The second argument specifies k (default=1)

            Another choice is 'softmax-avg', which computes gene weights by a
            softmax-type summation of loadings across the PCA/NMF columns,
            with each column's weight proportional to exp(QP wt), and only
            columns with QP weight > 1 being considered. *k is ignored here*. 

            Yet another choice is 'top-k-rank', which considers only the
            top-k PCA/NMF columns by QP-computed weight and computes the
            average rank of a gene across their loadings. The second argument specifies k (default=1)
 
            In all approaches, PCA loadings are first converted to
            absolute value, since PCA columns are unique up to a sign.

        :type k: int, >= 0

        :param k: 
            The number of PCA/NMF columns to average over, when affine_map_style =  top-k-loading or top-k-rank. 


        *returns* : a vector of floats, the same size as the primary dataset's dimensionality
"""
        if self._mode == "scale":
            return self._wts
        
        else:
            
            if affine_map_style == 'softmax-avg':
                w = np.exp(self._wts)  # exponentiation of wts so the highest-wt features really stand out
                w[ self._wts <= 1] = 0 # ignore features with wt <=1 (these were not up-weighted)
                w = w/np.sum(w) # normalize

                if self._params['decomposition_model'] == 'nmf':
                    df_comp = pd.DataFrame(self._decomp_mdl.components_)

                else: #in PCA, loadings are unique upto a sign, so when adding them up, we just take the magnitudes
                    df_comp = pd.DataFrame(self._decomp_mdl.components_).abs()

                feat_wts = (df_comp* w[:,None]).sum(axis=0).values
                return feat_wts

            
            elif affine_map_style == "top-k-rank":
                try:
                    assert k>0 and k < len(self._wts)
                except:
                    raise ValueError("""Incorrect "k" argument for 'top-k-rank'""")
                
                w = self._wts.copy()
                not_topk_idx = w.argsort()[:-k] 
                w[not_topk_idx] = 0 # set all but the top-k PCA/NMF column weights to zero
                w = w/np.sum(w) # normalize
            
                if self._params['decomposition_model'] == 'nmf':
                    df_comp = pd.DataFrame(self._decomp_mdl.components_).rank(axis=1, pct=True)

                else: #in PCA, loadings are unique upto a sign, so when adding them up, we just take the magnitudes
                    df_comp = pd.DataFrame(self._decomp_mdl.components_).abs().rank(axis=1, pct=True)

                feat_wts = (df_comp* w[:,None]).sum(axis=0).values
                return feat_wts

            
            elif affine_map_style == "top-k-loading":
                try:
                    assert k>0 and k < len(self._wts)
                except:
                    raise ValueError("""Incorrect "k" argument for 'top-k-loading'""")
                
                w = self._wts.copy()
                not_topk_idx = w.argsort()[:-k] 
                w[not_topk_idx] = 0 # set all but the top-k PCA/NMF column weights to zero
                w = w/np.sum(w) # normalize
            
                if self._params['decomposition_model'] == 'nmf':
                    df_comp = pd.DataFrame(self._decomp_mdl.components_)

                else: #in PCA, loadings are unique upto a sign, so when adding them up, we just take the magnitudes
                    df_comp = pd.DataFrame(self._decomp_mdl.components_).abs()

                feat_wts = (df_comp* w[:,None]).sum(axis=0).values
                return feat_wts

            else:
                raise ValueError(""" "style" needs to be one of 'top-k-loading' or 'softmax-avg' or 'top-k-rank'""")                



            



            
    ###################################################################
    ######## "private" methods below. Not that Python cares... ########
    ###################################################################
    def _getDistances(self, D, d0_in, G, nPointPairs):
        """
        Compute the various distances between point pairs in D
        D is a NxK Numpy type 2-d array, with N points, each of K dimensions
        d0_in could be None, in which D is used to compute distances for the original space.
           Otherwise, d0_in is Nxr (r>=1) array and original-space distances are computed from this. 
        G is list, with each entry a 3-tuple: (g_val, g_type, gamma_g)
           each 3-tuple corresponds to one dimension of side-information
             g_val is Nx1 Numpy type 1-d vector of values for the N points, in the same order as D
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
                schema_info ("Flag 201.82 ", g_val[i_u].shape, g_val[i_v].shape)
                dg = np.ravel(np.sum(np.power(g_val[i_u].astype(np.float64) - g_val[i_v].astype(np.float64),2), axis=1))
            elif g_type == "feature_vector_categorical":
                schema_info ("Flag 201.84 ", g_val[i_u].shape, g_val[i_v].shape)
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
        
        newmetric_sd = np.sqrt( np.matmul(np.matmul(np.reshape(w,(1,K)), P1), np.reshape(w,(K,1)))[0] / nPointPairs)
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
                schema_info ("Flag 110.50 crashed in _iterateQPLevel2. Trying to continue...", P1.size, q1.size, g1.size, max_w_wt, alpha, beta)
                schema_info(e)
                beta *= 0.5
                continue
            
            if soln["distcorr"] >= min_desired_oldnew_corr:
                solutionList.append((-soln["objval"], soln, param_settings))

            beta *= 0.5
        
        try:
            solutionList.sort(key=lambda v: v[0]) #find the highest score

            schema_debug ("Flag 110.60 beta: ", "NONE" if not solutionList else self._summarizeSoln(solutionList[0][1], solutionList[0][2]))
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
        
        schema_info ("Flag 102.40 Final solution: ", self._summarizeSoln(soln, free_params))
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
        
        self._wts = np.maximum(s,0) # shouldn't be <0 anyways, except for numerical issues
        self._wts = len(self._wts)*self._wts/np.sum(self._wts) #let's normalize so that avg wt = 1
        
        self._soln_info = dict(sl)
        print(' done.\n', end='', flush=True)
                

        
    def _fit_affine(self, d, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list):
        do_whiten = self._params.get("do_whiten",1)>0
        ncomp = self._params.get("num_top_components",None) #default is all
        model_type = self._params.get("decomposition_model","pca").lower()
        dx1 = d.copy()
        
        if self._decomp_mdl is None:
            print('Running change-of-basis transform ({0}, {1} components)...'.format(model_type, ncomp), end='', flush=True)

            if model_type=="pca":
                if not scipy.sparse.issparse(dx1):
                    self._decomp_mdl = sklearn.decomposition.PCA(n_components=ncomp, whiten=do_whiten)
                else:
                    self._decomp_mdl = sklearn.decomposition.TruncatedSVD(n_components=ncomp)
                    schema_warning("Using TruncatedSVD instead of PCA because input is a sparse matrix. do_whiten arguments will be ignored")
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._decomp_mdl.fit(dx1)

            elif model_type=="nmf":
                self._decomp_mdl = sklearn.decomposition.NMF(n_components=ncomp, random_state=0, alpha=0, l1_ratio=0)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._decomp_mdl.fit(dx1)
                
            self._orig_decomp_mdl = copy.deepcopy(self._decomp_mdl)
            print(' done.')
            
        else:
            print('Reusing previous change-of-basis transformation')
            self._decomp_mdl = copy.deepcopy(self._orig_decomp_mdl)

        if model_type=="pca":
            dx = self._decomp_mdl.transform(dx1)

        elif model_type=="nmf":
            W = self._decomp_mdl.transform(dx1)
            if do_whiten:
                H = self._decomp_mdl.components_
                
                wsd = np.ravel(np.sqrt(np.sum(W**2, axis=0))) #W.std(axis=0)
                W = W/wsd[None,:]
                self._decomp_mdl.components_ *= wsd[:,None]
                
                # hsd = np.sqrt(np.sum(H**2, axis=1))
                # H = H/hsd[:,None]
                # self._decomp_mdl.components_ = H
                # W = W*hsd[None,:]
                
            dx = W


            
        print('Running quadratic program...', end='', flush=True)
        s, sl = self._fit_helper(dx, d0, secondary_data_val_list, secondary_data_type_list, secondary_data_wt_list)

        self._wts = np.maximum(s,0) # shouldn't be <0 anyways, except for numerical issues
        self._wts = len(self._wts)*self._wts/np.sum(self._wts) #let's normalize so that avg wt = 1

        self._soln_info = dict(sl)
        print(' done.\n', end='', flush=True)


