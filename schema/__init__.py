
from .schema_qp import SchemaQP
from .schema_base_config import schema_loglevel

__all__ = ['SchemaQP', 'schema_loglevel']

from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

# https://julienharbulot.com/python-dynamical-import.html
#
# iterate through the modules in the current package
package_dir = Path(__file__).resolve().parent
for (_, module_name, _) in iter_modules([package_dir]):
    if 'datasets' in module_name:
        # import the module and iterate through its attributes
        module = import_module(f"{__name__}.{module_name}")
