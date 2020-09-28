|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/schema_learn.svg
   :target: https://pypi.org/project/schema_learn
.. |Docs| image:: https://readthedocs.com/projects/schema_learn/badge/?version=latest
   :target: https://schema-multimodal.readthedocs.io
..


Schema - Integrate Simultaneously-Assayed Single-Cell Modalities
=======================================

Schema is a Python library for the synthesis and integration of heterogeneous single-cell modalities.
It is designed for the case where the modalities have all been measured for the same cells simultaneously.
Some of the analyses that Schema facilitates are: cell type inference, identification of differentially expressed genes, batch effect incorporation. 
Additionally, Schema offers support for simultaneous incorporation of multiple modalities, batch effects, metadata (e.g., cell age).
It also enables more informative t-SNE & UMAP visualizations.

Schema is based on a metric learning approach and its Python-based implementation can efficiently process large datasets without the need of a GPU.

Read the documentation_.
If you'd like to contribute by opening an issue or creating a pull request at our Github page
If Schema is useful for your research, consider citing `bioRxiv (2019)`_.

.. _documentation: https://schema-multimodal.readthedocs.io 
.. _bioRxiv (2019): https://www.biorxiv.org/content/10.1101/834549v1
