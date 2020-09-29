|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/schema_learn.svg
   :target: https://pypi.org/project/schema_learn
.. |Docs| image:: https://readthedocs.org/projects/schema-multimodal/badge/?version=latest
   :target: https://schema-multimodal.readthedocs.io/en/latest/?badge=latest



Schema - Integrate Multiple Single-Cell Modalities
===================================================

Schema is a Python library for the synthesis and integration of heterogeneous single-cell modalities.
**It is designed for the case where the modalities have all been assayed for the same cells simultaneously.**
Some of the analyses that Schema can facilitate include: joint cell type inference across modalities, identification of differentially expressed genes, spatial transcriptomics analyses. 
Moreover, Schema offers support for simultaneous incorporation of multiple modalities, batch effects, metadata (e.g., cell age).


We think a pretty neat use of Schema is to infuse scRNA-seq data with information from additional modalities, allowing the creation of more informative t-SNE & UMAP visualizations.

Schema is based on a metric learning approach and formulates the modality-synthesis problem as a quadratic programming problem. Its Python-based implementation can efficiently process large datasets without the need of a GPU.

Read the documentation_.
If you'd like to contribute by opening an issue or creating a pull request at our `Github page`_
If Schema is useful for your research, consider citing `bioRxiv (2019)`_.

.. _documentation: https://schema-multimodal.readthedocs.io 
.. _bioRxiv (2019): https://www.biorxiv.org/content/10.1101/834549v1
.. _Github page: https://github.com/rs239/schema
