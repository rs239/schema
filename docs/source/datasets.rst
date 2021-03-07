Datasets
=========

Aging Drosophila Brain
~~~~~~~~~~~~~~~~~~~~~~

This is sourced from Davie et al. (Cell 2018, GSE 107451) and contains scRNA-seq data from a collection of fly brain cells along with each cell's age (in days). It is a useful dataset to explore the use of Schema for a common scenario in multi-modal integration:  RNA-seq data against a 1-dimensional secondary modality.

.. code-block:: Python

   import schema
   adata = schema.datasets.fly_brain()


Next Datasets
~~~~~~~~~~~~~

.. code-block:: Python

    from schema import SchemaQP

Coming soon: example datasets from Sci-CAR,  10x ATAC+scRNA, SlideSeq etc. 




