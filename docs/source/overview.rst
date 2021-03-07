Overview
========


Schema is a general algorithm for integrating heterogeneous data
modalities. While it has been specially designed for multi-modal
single-cell biological datasets, it should work in other multi-modal
contexts too.




Basic Usage
~~~~~~~~~~~

.. code-block:: Python

    from schema import SchemaQP


:Example: Correlate gene expression with developmental stage. We demonstrate use with Anndata objects here.
.. code-block:: Python

    sqp = SchemaQP() # initialize with default params (min_corr = 0.99)
    mod_X = sqp.fit_transform( adata.X, [ adata.obs['stage'] ]) # correlate the gene expression with the 'stage' parameter
    gene_wts = sqp.feature_weights() # get a ranking of gene wts important to the correlation


:Example: Correlate gene expression with three secondary modalities.
.. code-block:: Python

    sqp = SchemaQP(min_corr = 0.9) # lower than the default, allowing greater distortion of the primary modality 
    sqp.fit( adata.X,    
                 [ adata.obs['col1'], adata.obs['col2'], adata.obsm['Matrix1'] ], 
                 [ "categorical", "numeric", "feature_vector"]) # data types of the three modalities
    mod_X = sqp.transform( adata.X) # transform
    gene_wts = sqp.feature_weights() # get gene importances


:Example: Correlate gene expression 1) positively with ATAC-Seq data and 2) negatively with Batch information::
.. code-block:: Python

    atac_30d = sklearn.decomposition.TruncatedSVD(50).fit_transform( atac_cnts_sp_matrix)
    sqp = SchemaQP(min_corr=0.9)
    # df is a pd.DataFrame, srs is a pd.Series, -1 means try to disagree
    mod_X = sqp.fit_transform( df_gene_exp, # gene expression dataframe
                               [ atac_30d, batch_id],  # batch_info can be a Pandas Series or numpy array
                               [ 'feature_vector', 'categorical'], 
                               [ 1, -1]) # maximize combination of (agreement with ATAC-seq + disagreement with batch_id)
    gene_wts = sqp.feature_weights() # get gene importances



Schema is described in the paper “Schema: metric learning enables
interpretable synthesis of heterogeneous single-cell modalities" 
(http://doi.org/10.1101/834549)

Source code available at: https://github.com/rs239/schema
