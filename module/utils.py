import inspect
import warnings
import importlib.util
from typing import Any, Dict, Optional, Union

import sys

from packaging import version
import torch

def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since diffusers'"
                f" version {__version__} is >= {version_name}"
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=2)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(f"{function} in {filename} line {line_number-1} got an unexpected keyword argument `{key}`")

    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values

try:
    from torch._dynamo import allow_in_graph as maybe_allow_in_graph
except (ImportError, ModuleNotFoundError):

    def maybe_allow_in_graph(cls):
        return cls


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    import torch

    if version.Version(torch.__version__) < version.Version("1.12"):
        raise ValueError("PyTorch should be >= 1.12")
    print(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False

def is_xformers_available():
    return _xformers_available

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob