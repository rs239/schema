|PyPI| |Docs|

.. |PyPI| image:: https://img.shields.io/pypi/v/schema_learn.svg
   :target: https://pypi.org/project/schema_learn
.. |Docs| image:: https://readthedocs.org/projects/schema-multimodal/badge/?version=latest
   :target: https://schema-multimodal.readthedocs.io/en/latest/?badge=latest



Schema - Analyze and Visualize Multimodal Single-Cell Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Schema is a Python library for the synthesis and integration of heterogeneous single-cell modalities.
**It is designed for the case where the modalities have all been assayed for the same cells simultaneously.**
Here are some of the analyses that you can do with Schema:

  - infer cell types jointly across modalities.
  - perform spatial transcriptomic analyses to identify differntially-expressed genes in cells that display a specific spatial characteristic.
  - create informative t-SNE & UMAP visualizations of multimodal data by infusing information from other modalities into scRNA-seq data.
    
Schema offers support for the incorporation of more than two modalities and can also simultaneously handle batch effects and metadata (e.g., cell age).


Schema is based on a metric learning approach and formulates the modality-synthesis problem as a quadratic programming problem. Its Python-based implementation can efficiently process large datasets without the need of a GPU.

Read the documentation_.
We encourage you to report issues at our `Github page`_ ; you can also create pull reports there to contribute your enhancements.
If Schema is useful for your research, please consider citing `bioRxiv (2019)`_.

.. _documentation: https://schema-multimodal.readthedocs.io/en/latest/overview.html
.. _bioRxiv (2019): https://www.biorxiv.org/content/10.1101/834549v1
.. _Github page: https://github.com/rs239/schema
