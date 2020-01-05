# Schema

Schema is a general algorithm for integrating heterogeneous data
modalities. It has been specially designed for multi-modal
single-cell biological datasets, but should work in other contexts too.
This version is based on a Quadratic Programming framework.


It is described in the paper
["*Schema: A general framework for integrating heterogeneous single-cell modalities*"](https://www.biorxiv.org/content/10.1101/834549v1).



The module provides a class SchemaQP that offers a sklearn type fit+transform API for affine
transformations of input datasets such that the transformed data is in agreement
with all the input datasets.

## Getting Started
The examples provided here are also available in the examples/Schema_demo.ipynb notebook

### Installation
```
pip install schema_learn
```

### Schema: A simple example

For the examples below, you'll also need scanpy (`pip install scanpy`).
We use `fast_tsne` below for visualization, but feel free to use your favorite tool.

#### Sample data

The data in the examples below is from the paper below; we thank the authors for making it available:
  * Tasic et al. [*Shared and distinct transcriptomic cell types across neocortical areas*.](https://www.nature.com/articles/s41586-018-0654-5) Nature. 2018 Nov;563(7729):72-78. doi:10.1038/s41586-018-0654-5

We make available a processed subset of the data for demonstration and analysis.
Linux shell commands to get this data:
```
wget http://schema.csail.mit.edu/datasets/Schema_demo_Tasic2018.h5ad.gz
gunzip Schema_demo_Tasic2018.h5ad.gz
```

In Python, set the `DATASET_DIR` variable to the folder containing this file.

The processing of raw data here broadly followed the steps in Kobak & Berens
  * https://www.biorxiv.org/content/10.1101/453449v1

The gene expression data has been count-normalized and log-transformed. Load with the commands
```python
import scanpy as sc
adata = sc.read(DATASET_DIR + "/" + "Schema_demo_Tasic2018.h5ad")
```

#### Sample Schema usage

Import Schema as:
```python
from schema import SchemaQP
afx = SchemaQP(0.75) # min_desired_corr is the only required argument.

dx_pca = afx.fit_transform(adata.X, # primary dataset
                           [adata.obs["class"].values], # just one secondary dataset
                           ['categorical'] # has labels, i.e., is a categorical datatype
                          )
```
This uses PCA as the change-of-basis transform; requires a min corr of 0.75 between the
primary dataset (gene expression) and the transformed dataset; and maximizes
correlation between the primary dataset and the secondary dataset, supercluster
(i.e. higher-level clusters) labels produced during Tasic et al.'s hierarchical clustering.


### More Schema examples
  * In all of what follows, the primary dataset is gene expression. The secondary datasets are 1) cluster IDs; and/or 2) cell-type "class" variables which correspond to superclusters (i.e. higher-level clusters) in the Tasic et al. paper.



#### With NMF (Non-negative Matrix Factorization) as change-of-basis, a different min_desired_corr, and two secondary datasets


```python
afx = SchemaQP(0.6, params= {"decomposition_model": "nmf", "num_top_components": 50})

dx_nmf = afx.fit_transform(adata.X,
                           [adata.obs["class"].values, adata.obs.cluster_id.values], # two secondary datasets
                           ['categorical', 'categorical'], # both are labels
                           [10, 1] # relative wts
                     )
```

#### Now let's do something unusual. Perturb the data so it *disagrees* with cluster ids


```python
afx = SchemaQP(0.97, # Notice that we bumped up the min_desired_corr so the perturbation is limited
               params = {"decomposition_model": "nmf", "num_top_components": 50})

dx_perturb = afx.fit_transform(adata.X,
                           [adata.obs.cluster_id.values], # could have used both secondary datasets, but one's fine here
                           ['categorical'],
                           [-1] # This is key: we are putting a negative wt on the correlation
                          )
```


#### Recommendations for parameter settings
  * `min_desired_corr` and `w_max_to_avg` are the names for the hyperparameters $s_1$ and $\bar{w}$ from our paper
  * *min_desired_corr*: at first, you should try a range of values for `min_desired_corr` (e.g., 0.99, 0.90, 0.50). This will give you a sense of what might work well for your data; after this, you can progressively narrow down your range. In typical use-cases, high `min_desired_corr` values (> 0.80) work best.
  * *w_max_to_avg*: start by keeping this constraint very loose. This ensures that `min_desired_corr` remains the binding constraint. Later, as you get a better sense for `min_desired_corr` values, you can experiment with this too. A value of 100 is pretty high and should work well in the beginning.



#### tSNE plots of the baseline and Schema transforms

```python
fig = plt.figure(constrained_layout=True, figsize=(8,2), dpi=300)
tmps = {}
for i,p in enumerate([("Original", adata.X),
                      ("PCA1 (pos corr)", dx_pca),
                      ("NMF (pos corr)", dx_nmf),
                      ("Perturb (neg corr)", dx_perturb)
                     ]):
    titlestr, dx1 = p
    ax = fig.add_subplot(1,4,i+1, frameon=False)
    tmps[titlestr] = dy = fast_tsne(dx1, seed=42)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')
    ax.scatter(dy[:,0], dy[:,1], s=1, color=adata.obs['cluster_color'])
    ax.set_title(titlestr)
    ax.axis("off")
```



## API

### Constructor
Initializes the `SchemaQP` object

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


`mode`: {`'affine'`, `'scale'`}, optional (default: `'affine'`)

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


### Fit
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


`secondary_data_type_list`: `list` of `string`s, each value in {'numeric','feature_vector','categorical'}

    The list's length should match the length of secondary_data_val_list

    * 'numeric' means you're giving one floating-pt value for each obs.
          The default distance measure is L2:  (point1-point2)**2
    * 'feature_vector' means you're giving some multi-dimensional representation for each obs.
          The default distance measure is L2: sum((point1-point2)**2)
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


### Transform
Given a dataset `d`, apply the fitted transform to it


#### Parameters

`d`:  a numpy 2-d array with same number of columns as primary dataset `d` in the fit(...)

    The rows are observations (e.g., cells) and the cols are variables (e.g., gene expression).


#### Returns

 a 2-d numpy array with the same shape as `d`
