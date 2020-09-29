|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/schema_learn.svg
   :target: https://pypi.org/project/schema_learn
.. |Docs| image:: https://readthedocs.org/projects/schema-multimodal/badge/?version=latest
   :target: https://schema-multimodal.readthedocs.io/en/latest/?badge=latest



Schema - Analyze and Visualize Multi-modal Single-Cell Data
===========================================================

Schema is a Python library for the synthesis and integration of heterogeneous single-cell modalities.
**It is designed for the case where the modalities have all been assayed for the same cells simultaneously.**
Some of the analyses that you can do with Schema include:

  - infer cell types jointly across modalities.
  - identify genes whose expression correlates with the cell state as measured by other modalities.
  - perform spatial transcriptomic analyses to identify differntially-expressed genes and cell types that display a specific spatial characteristic.
  - create informative visualizations of multi-modal data by infusing information from other modalities into scRNA-seq data, and plotting the synthesized data with t-SNE or UMAP.
    
Schema offers support for the incorporation of more than two modalities and can also simultaneously handle batch effects and metadata (e.g., cell age).


Schema is based on a metric learning approach and formulates the modality-synthesis problem as a quadratic programming problem. Its Python-based implementation can efficiently process large datasets without the need of a GPU.

Read the documentation_.
We encourage you to report issues and create pull reports to contribute your enhancements at our `Github page`_.
If Schema is useful for your research, please consider citing `bioRxiv (2019)`_.


.. _documentation: https://schema-multimodal.readthedocs.io 
.. _bioRxiv (2019): https://www.biorxiv.org/content/10.1101/834549v1
.. _Github page: https://github.com/rs239/schema
