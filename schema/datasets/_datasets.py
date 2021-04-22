#!/usr/bin/env python


###################################################################
## Primary Author:  Rohit Singh rsingh@alum.mit.edu
## Co-Authors: Ashwin Narayan, Brian Hie {ashwinn,brianhie}@mit.edu
## License: MIT
## Repository:  http://github.io/rs239/schema
###################################################################

import sys, copy, os, warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scanpy


# #### local directory imports ####
# oldpath = copy.copy(sys.path)
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))+"/../")

# from schema_base_config import *

# sys.path = copy.copy(oldpath)
# ####




def fly_brain():
    """ Anndata object containing scRNA-seq data of the ageing Drosophila brain (GSE107451, Davie et al., Cell 2018)
"""
    
    adata = scanpy.read("datasets/Davie_fly_brain.h5", backup_url="http://schema.csail.mit.edu/datasets/Davie_fly_brain.h5")
    return adata


def scicar_mouse_kidney():
    """ Anndata object containing scRNA-seq+ATAC-seq data of mouse kidney cells from the Sci-CAR study (GSE117089, Cao et al., Science 2018)
"""
    
    adata = scanpy.read("datasets/Cao_mouse_kidney.h5", backup_url="http://schema.csail.mit.edu/datasets/Cao_mouse_kidney.h5")
    return adata

