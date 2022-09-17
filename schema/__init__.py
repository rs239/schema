
from .schema_qp import SchemaQP
from .schema_base_config import schema_loglevel

__all__ = ['SchemaQP', 'schema_loglevel']


import pkgutil, pathlib, importlib

# from pkgutil import iter_modules
# from pathlib import Path
# from importlib import import_module

# https://julienharbulot.com/python-dynamical-import.html
# iterate through the modules in the current package
#
package_dir = str(pathlib.Path(__file__).resolve().parent)

for (_, module_name, _) in pkgutil.iter_modules([package_dir]):
    if 'datasets' in module_name:
        module = importlib.import_module(f"{__name__}.{module_name}")
