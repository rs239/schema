from schema import SchemaQP
from anndata import AnnData
import numpy as np
import scanpy as sc

from .process import load_names

def load_meta(fname):
    age, strain = [], []
    with open(fname) as f:
        f.readline() # Consume header.
        for line in f:
            fields = line.rstrip().split()
            age.append(int(fields[4]))
            strain.append(fields[3])
    return np.array(age), np.array(strain)

if __name__ == '__main__':
    [ X ], [ genes ], _ = load_names([ 'data/fly_brain/GSE107451' ], norm=False)

    age, strain = load_meta('data/fly_brain/GSE107451/annotation.tsv')

    # Only analyze wild-type strain.
    adata = AnnData(X[strain == 'DGRP-551'])
    adata.var['gene_symbols'] = genes
    adata.obs['age'] = age[strain == 'DGRP-551']

    # No Schema transformation.

    sc.pp.pca(adata)
    sc.tl.tsne(adata, n_pcs=50)
    sc.pl.tsne(adata, color='age', color_map='coolwarm',
               save='_flybrain_regular.png')

    sc.pp.neighbors(adata, n_neighbors=15)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color='age', color_map='coolwarm',
               save='_flybrain_regular.png')

    # Schema transformation to include age.

    schema_corrs = [ 0.9999, 0.999, 0.99, 0.9, 0.7, 0.5 ]

    for schema_corr in schema_corrs:

        sqp = SchemaQP(
            min_desired_corr=schema_corr,
            w_max_to_avg=100,
            params={
                'decomposition_model': 'nmf',
                'num_top_components': 20,
            },
        )

        X = sqp.fit_transform(
            adata.X,
            [ adata.obs['age'].values, ],
            [ 'numeric', ],
            [ 1, ]
        )

        sdata = AnnData(X)
        sdata.obs['age'] = age[strain == 'DGRP-551']

        sc.tl.tsne(sdata)
        sc.pl.tsne(sdata, color='age', color_map='coolwarm',
                   save='_flybrain_schema_corr{}_w100.png'.format(schema_corr))

        sc.pp.neighbors(sdata, n_neighbors=15)
        sc.tl.umap(sdata)
        sc.pl.umap(sdata, color='age', color_map='coolwarm',
                   save='_flybrain_schema{}_w100.png'.format(schema_corr))
