
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_optimizer']

#======================================
# Optimizer assignment  
#======================================

optimizer_avaiable = {
    'Adam'.lower(): torch.optim.Adam,
    'AMSgrad'.lower(): torch.optim.Adam, 
    }
optimizer_argumens = {
    'Adam'.lower(): {},
    'AMSgrad'.lower(): {
        'amsgrad': True}, 
    }

def get_optimizer(
    trainer_optimizer: Union[str, object],
    model_parameter: Optional[List] = None,
    trainer_optimizer_args: Optional[Dict[str, Any]] = {},
):
    """
    Optimizer selection
    
    Parameters
    ----------
        
    trainer_optimizer: (str, object)
        If name is a str than it checks for the corresponding optimizer
        and return the function object.
        The input will be given if it is already a callable object.
    model_parameter: list, optional
        NNP model trainable parameter to optimize by optimizer.
        Optional if 'trainer_optimizer' is already a torch optimizer object
    trainer_optimizer_args: dict, optional
        Additional optimizer parameter.
        Optional if 'trainer_optimizer' is already a torch optimizer object
            
    Returns
    -------
    object
        Optimizer function
    """
    
    # Select calculator model
    if utils.is_string(trainer_optimizer):
        
        # Check required input for this case
        if model_parameter is None:
            raise SyntaxError(
                f"In case of defining 'trainer_optimizer' as string, the " +
                f"optional input parameter 'model_parameter' must be defined!")
        
        # Check optimizer availability 
        if trainer_optimizer.lower() in optimizer_avaiable.keys():

            if trainer_optimizer.lower() in optimizer_argumens.keys():
                trainer_optimizer_args.update(
                    optimizer_argumens[trainer_optimizer.lower()])
            
            try:
                return optimizer_avaiable[trainer_optimizer.lower()](
                    params=model_parameter,
                    **trainer_optimizer_args)
            except TypeError as error:
                logger.error(error)
                raise TypeError(
                    f"Optimizer '{trainer_optimizer}' does not accept " +
                    f"arguments in 'trainer_optimizer_args'")

        else:
            
            raise ValueError(
                f"Optimizer class '{trainer_optimizer}' is not valid!" +
                f"Choose from:\n" +
                str(optimizer_avaiable.keys()))

    else:
        
        return trainer_optimizer
