#!/usr/bin/env python


###################################################################
## Primary Author:  Rohit Singh rsingh@alum.mit.edu
## Co-Authors: Ashwin Narayan, Brian Hie {ashwinn,brianhie}@mit.edu
## License: MIT
## Repository:  http://github.io/rs239/schema
###################################################################

import sys, copy, os
import scanpy as sc

#### local directory imports ####
oldpath = copy.copy(sys.path)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schema_base_config import *

sys.path = copy.copy(oldpath)
####




def fly_brain():
    """ Anndata object containing scRNA-seq data of the aging Drosophila brain (GSE107451, Davie et al., Cell 2018)
"""
    
    adata = sc.read("datasets/Davie_fly_brain.h5", backup_url="http://schema.csail.mit.edu/datasets/Davie_fly_brain.h5")
    return adata
