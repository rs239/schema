#!/usr/bin/env python

# author: Rohit Singh
## CREDITS: modeled on scanpy's documentation ( http://scanpy.readthedocs.io)

import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib  # noqa

# Don’t use tkinter agg when importing scanpy → … → matplotlib
matplotlib.use('agg')

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE.parent), str(HERE / 'extensions')]

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

