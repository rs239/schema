#!/usr/bin/env python

# author: Rohit Singh
## CREDITS: modeled on scanpy's documentation ( http://scanpy.readthedocs.io)

import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib  
matplotlib.use('agg')

sys.path[:0] = [os.path.dirname(os.path.absname(__file__)) + "/../../schema/"] 

import schema

templates_path = ['_templates']

project = 'Schema'
copyright = '2020, Rohit Singh, Brian Hie, Ashwin Narayan & Bonnie Berger. '
source_suffix = '.rst'
master_doc = 'index'

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only':  True,
}

html_static_path = ['_static']
html_logo = '_static/Schema-webpage-logo-2-blue.png'

