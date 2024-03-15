import os
import sys
import logging
import datetime
from typing import Optional, List, Dict, Tuple, Union, Callable

import numpy as np 

import torch
#import pytorch_lightning as pl
#from pytorch_lightning.accelerators import GPUAccelerator #TODO

from . import settings
from . import utils
from . import data
#from . import sample
from . import model
#from . import train
#from . import interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Asparagus']

class Asparagus():
    """
    Asparagus main class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    kwargs: dict, optional, default {}
        Additional model keyword input parameter

    Returns
    -------
    object
        Main Asparagus object to direct tasks.
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):

        super().__init__()

        #############################
        # # # Check Model Input # # #
        #############################

        # Initialize model parameter configuration dictionary
        # Keyword arguments overwrite entries in the configuration dictionary
        self.config = settings.get_config(
            config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        self.config.check(
            check_default=True,
            check_dtype=True,
            )

        ###########################
        # # # Class Parameter # # #
        ###########################

        # DataContainer of reference data
        self.data_container = None
        # Model calculator
        self.model_calculator = None
        # Model trainer
        self.trainer = None
        # Model testing
        self.tester = None

        return

    def __str__(self):
        """
        Return class descriptor
        """
        return "Asparagus Main"

    def __getitem__(self, args):
        """
        Return item(s) from configuration dictionary
        """
        return self.config.get(args)

    def get(self, args):
        """
        Return item(s) from configuration dictionary
        """
        return self.config.get(args)

    def set_DataContainer(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize DataContainer and set as class variable

        Parameter:
        ----------
        config: (str, dict, object), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)
        **kwargs: dict, optional, default {}
            Additional model keyword input parameter
        """
        
        # Check input
        if config is None:
            config = self.config
        print('kwargs', kwargs)
        # Initialize DataContainer
        self.data_container = data.DataContainer(
            config,
            config_file,
            **kwargs
            )
        
        return
    
    def get_DataContainer(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ) -> data.DataContainer:
        """
        Initialize and return DataContainer.

        Parameter:
        ----------
        config: (str, dict, object), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)
        **kwargs: dict, optional, default {}
            Additional model keyword input parameter
        """
        
        # Check input
        if config is None:
            config = self.config

        return data.DataContainer(
            config,
            config_file,
            **kwargs
            )

    def get_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[int] = None,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Return calculator model class object

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_checkpoint: int, optional, default None
            If None, load best model checkpoint. Otherwise define a checkpoint
            index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """

        ########################################
        # # # Check Model Calculator Input # # #
        ########################################

        # Assign model parameter configuration library
        if config is None:
            config_model = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_model = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_model.check(
            check_default=True,
            check_dtype=True,
            )

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign model calculator
        model_calculator = self.model.get_model_calculator(
            config_model,
            **kwargs)
        
        # Add calculator info to configuration dictionary
        if hasattr(model_calculator, "get_info"):
            config_model.update(
                model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_model, **kwargs)
        if model_checkpoint is None:
            checkpoint = filemanager.load_checkpoint(best=True)
        elif utils.is_integer(model_checkpoint):
            checkpoint = filemanager.load_checkpoint(
                num_checkpoint=model_checkpoint)
        else:
            raise ValueError(
                "Input 'model_checkpoint' must be either None to load best "
                + "model checkpoint or an integer of a respective checkpoint "
                + "file.")
        self.model_calculator.load_state_dict(checkpoint['model_state_dict'])
        
        return self.model_calculator    
