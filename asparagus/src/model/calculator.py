
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch
#import pytorch_lightning as pl

from .. import settings
from .. import utils

from .physnet import Calculator_PhysNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_calculator']


# ======================================
# Calculator Assignment
# ======================================

calculator_avaiable = {
    'PhysNet'.lower(): Calculator_PhysNet,
    'PhysNet_original'.lower(): Calculator_PhysNet,
    }

def get_calculator(
    config: Optional[Union[str, dict, object]] = None,
    model_directory: Optional[str] = None,
    model_num_threads: Optional[int] = None,
    model_type: Optional[str] = None,
    **kwargs
):
    """
    Calculator selection

    Parameters
    ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        model_directory: str, optional, default None
            Calculator model directory for file management
        model_num_threads: int, optional, default None
            Maximum number of threads for Pytorch. If None, use Pytorch
            default number of threads.
        model_type: str, optional, default 'PhysNet'
            Calculator model type of the NN potential
            e.g. 'PhysNetRBF'
        **kwargs: dict, optional
            Additional arguments for parameter initialization

    Returns
    -------
        callable object
            Calculator object to predict properties from input
    """

    # Check configuration object
    config = settings.get_config(config)

    # Check input parameter, set default values if necessary and
    # update the configuration dictionary
    config_update = {}
    for arg, item in locals().items():
        
        # Skip 'config' argument and possibly more
        if arg in ['self', 'config', 'config_update', 'kwargs', '__class__']:
            continue
        
        # Take argument from global configuration dictionary if not defined
        # directly
        if item is None:
            item = config.get(arg)
        
        # Set default value if the argument is not defined (None)
        if arg in settings._default_args.keys() and item is None:
            item = settings._default_args[arg]
        
        # Check datatype of defined arguments
        if arg in settings._dtypes_args.keys():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)
        
        # Append to update dictionary
        config_update[arg] = item
        
    # Update global configuration dictionary
    config.update(config_update)

    # Calculator model type assignment
    if config.get('model_type') is None:
        config['model_type'] = settings._default_calculator_model
    model_type = config['model_type']

    # Set model calculator number of threads
    if config.get('model_num_threads') is not None:
        torch.set_num_threads(config.get('model_num_threads'))

    # Select calculator model
    if model_type.lower() in calculator_avaiable.keys():

        return calculator_avaiable[model_type.lower()](
            config,
            **kwargs)

    else:
        
        raise ValueError(
            f"Calculator model type '{model_type}' is not valid!" +
            f"Choose from:\n" +
            str(calculator_avaiable.keys()))
    
