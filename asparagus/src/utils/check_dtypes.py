
import numpy as np

import logging
from typing import Callable

import torch
import ase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- ** Checking data types ** ---------------

# Numeric data types
dint_all = (
    int, np.int16, np.int32, np.int64, 
    torch.int, torch.int16, torch.int32, torch.int64)
dflt_all = (
    float, np.float16, np.float32, np.float64,
    torch.float, torch.float16, torch.float32, torch.float64)
dnum_all = dint_all + dflt_all

# Array data types
darr_all = (tuple, list, np.ndarray, torch.Tensor)

# Bool data types
dbool_all = (bool, np.bool_)

def is_None(x, verbose=False):

    '''
    Check if the input is None

    Parameters
    ----------
    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    '''

    if verbose:
        return (x is None), type(x), "NoneType"
    else:
        return (x is None)

def is_string(x, verbose=False):
    '''
    Check if the input is a string

    Parameters
    ----------
    x: Any
    verbose: bool
        If True, return the type of the input and the expected type


    '''
    if verbose:
        return isinstance(x, str), type(x), "str"
    else:
        return isinstance(x, str)

def is_bool(x, verbose=False):
    '''

    Check if the input is a bool

    Parameters
    ----------
    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    '''
    if verbose:
        return isinstance(x, dbool_all), type(x), "bool"
    else:
        return isinstance(x, dbool_all)

def is_numeric(x, verbose=False):
    '''
    Check if the input is a numeric type for torch tensors

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if isinstance(x, torch.Tensor):
        if len(x.shape)==0:
            result = ((x.dtype in dnum_all), type(x), str(dnum_all))
        else:
            result = (False, type(x), str(dnum_all))
    else:
        result = (
            (type(x) in dnum_all and not is_bool(x)), type(x), str(dnum_all))

    if verbose:
        return result
    else:
        return result[0]

def is_integer(x, verbose=False):
    '''
    Check if the input is an integer type for torch tensors

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if isinstance(x, torch.Tensor):
        if len(x.shape)==0:
            result = ((x.dtype in dint_all), type(x), str(dint_all))
        else:
            result = (False, type(x), str(dint_all))
    else:
        result = (
            (type(x) in dint_all and not is_bool(x)), type(x), str(dint_all))

    if verbose:
        return result
    else:
        return result[0]

def is_callable(x, verbose=False):
    '''
    Check if the input is a callable object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type
    Returns
    -------

    '''
    if verbose:
        return isinstance(x, Callable), type(x), "callable object"
    else:
        return isinstance(x, Callable)

def is_object(x, verbose=False):
    '''
    Check if the input is an object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return isinstance(x, object), type(x), "object"
    else:
        return isinstance(x, object)

def is_dictionary(x, verbose=False):
    '''
    Check if the input is a dictionary

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return isinstance(x, dict), type(x), "dict"
    else:
        return isinstance(x, dict)

def is_array_like(x, verbose=False):
    '''
    Check if the input is an array-like object also checks for inhomogeneity
    of the array object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        # Test for inhomogeneity of the array object 
        try:
            _ = np.asarray(x)
        except ValueError as error:
            logger.warning(error)
        return isinstance(x, darr_all), type(x), str(darr_all)
    else:
        return isinstance(x, darr_all)

def is_numeric_array(x, verbose=False):
    '''
    Check if the input is a numeric array-like object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=float)
            result = (
                True,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[{dnum_all}]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[{dnum_all}]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]", 
            f"({darr_all})[{dnum_all}]")
    else:
        result = (
            False, 
            f"{type(x)}",
            f"({darr_all})[{dnum_all}]")

    if verbose:
        return result
    else:
        return result[0]


def is_integer_array(x, verbose=False):
    '''
    Check if the input is an integer array-like object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if is_numeric_array(x):
        if (np.asarray(x, dtype = float) == np.asarray(x, dtype = int)).all():
            result = (
                True,
                f"{type(x)}",
                f"({darr_all})[{dint_all}]")
    elif is_array_like(x):
        result = (
            True, 
            f"({type(x)})[empty]",
            f"({darr_all})[{dint_all}]")
    else:
        result = (
            False, 
            f"({type(x)})[{type(x[0])}]",
            f"({darr_all})[{dint_all}]")

    if verbose:
        return result
    else:
        return result[0]

def is_string_array(x, verbose=False):
    '''
    Check if the input is a string array-like object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=str)
            result = (
                True,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[str]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[str]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[str]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[str]")

    if verbose:
        return result
    else:
        return result[0]

def is_bool_array(x, verbose=False):
    '''
    Check if the input is a boolean array-like object. Wrapper for is_boolean_array.

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    return is_boolean_array(x, verbose=verbose)

def is_boolean_array(x, verbose=False):
    '''
    Check if the input is a boolean array-like object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            np.asarray(x, dtype=bool)
            result = (
                True,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[bool]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[bool]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[bool]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[bool]")

    if verbose:
        return result
    else:
        return result[0]

def is_None_array(x, verbose=False):
    '''
    Check if the input is a None array-like object.

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if is_array_like(x) and np.asarray(x).size > 0:
        try:
            result = (
                (np.array(x) is None).all(),
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[None]")
        except (ValueError, TypeError):
            result = (
                False,
                f"({type(x)})[{type(x[0])}]",
                f"({darr_all})[None]")
    elif is_array_like(x):
        result = (
            True,
            f"({type(x)})[empty]",
            f"({darr_all})[None]")
    else:
        result = (
            False,
            f"{type(x)}",
            f"({darr_all})[None]")

    if verbose:
        return result
    else:
        return result[0]

def is_grad_enabled(x, verbose=False):
    '''
    Check if the input is a torch tensor with gradient enabled

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return (
            isinstance(x, torch.autograd.Variable), type(x),
            f"Gradient for {x} is active")
    else:
        return isinstance(x, torch.autograd.Variable)

def in_cuda(x, verbose=False):
    '''

    Check if the input is a torch tensor in CUDA

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return x.is_cuda, type(x), "Tensor is in CUDA"
    else:
        return x.is_cuda

def is_attached(x, verbose=False):
    '''

    Check if the input is a torch tensor attached to the computational graph

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return x.is_leaf, type(x), "Tensor is attached"
    else:
        return x.is_leaf

def detach_tensor(x):
    '''

    Detach a torch tensor from the computational graph

    Parameters
    ----------

    x: Any

    Returns
    -------

    '''
    if in_cuda(x):
        x.cpu()
        x.detach().numpy()
    else:
        x.detach().numpy()
    return x

def is_ase_atoms(x, verbose=False):
    '''

    Check if the input is an ASE atoms object

    Parameters
    ----------

    x: Any
    verbose: bool
        If True, return the type of the input and the expected type

    Returns
    -------

    '''
    if verbose:
        return isinstance(x, ase.atoms.Atoms), type(x), "ASE atoms object"
    else:
        return isinstance(x, ase.atoms.Atoms)
