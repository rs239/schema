#!/usr/bin/env python

import logging

schema_loglevel = logging.WARNING #WARNING


def schema_debug(*args, **kwargs):
    if schema_loglevel <= logging.DEBUG: print("DEBUG: ", *args, **kwargs)

def schema_info(*args, **kwargs):
    if schema_loglevel <= logging.INFO: print("INFO: ", *args, **kwargs)

def schema_warning(*args, **kwargs):
    if schema_loglevel <= logging.WARNING: print("WARNING: ", *args, **kwargs)

def schema_error(*args, **kwargs):
    if schema_loglevel <= logging.ERROR: print("ERROR: ", *args, **kwargs)

    
########## for maintenance ###################
# def noop(*args, **kwargs):
#     pass
#
# logging.info = print
# logging.debug = noop
##############################################
