Example Usage
=======


*Note*: The code snippets below show how Schema could be used for hypothetical datasets. In `Visualization`_, we describe a worked example where we also provide the dataset to try things on. We are working to add more datasets.


**Example** Correlate gene expression 1) positively with ATAC-Seq data and 2) negatively with Batch information.
  
.. code-block:: Python

    atac_50d = sklearn.decomposition.TruncatedSVD(50).fit_transform( atac_cnts_sp_matrix)
    
    sqp = SchemaQP(min_corr=0.9)
    
    # df is a pd.DataFrame, srs is a pd.Series, -1 means try to disagree
    mod_X = sqp.fit_transform( df_gene_exp, # gene expression dataframe: rows=cells, cols=genes
                               [ atac_50d, batch_id],  # batch_info can be a pd.Series or np.array. rows=cells
                               [ 'feature_vector', 'categorical'], 
                               [ 1, -1]) # maximize combination of (agreement with ATAC-seq + disagreement with batch_id)
			       
    gene_wts = sqp.feature_weights() # get gene importances


 
**Example** Correlate gene expression with three secondary modalities.

.. code-block:: Python

    sqp = SchemaQP(min_corr = 0.9) # lower than the default, allowing greater distortion of the primary modality 
    sqp.fit( adata.X,    
             [ adata.obs['col1'], adata.obs['col2'], adata.obsm['Matrix1'] ], 
             [ "categorical", "numeric", "feature_vector"]) # data types of the three modalities
    mod_X = sqp.transform( adata.X) # transform
    gene_wts = sqp.feature_weights() # get gene importances



.. _Visualization: https://schema-multimodal.readthedocs.io/en/latest/visualization/index.html#ageing-fly-brain
