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
from . import model
from . import train

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
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Get configuration file path
        self.config_file = self.config.get('config_file')

        # Print Asparagus header
        utils.header(self.config_file)

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

    def set_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_calculator: Optional[torch.nn.Module] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ):
        """
        Set and, eventually, initialize the calculator model class object

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_calculator: torch.nn.Module, optional, default None
            Model calculator object to assign as class model calculator.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: int, optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """
        
        # Assign model parameter configuration library
        if config is None:
            config_model = settings.get_config(
                self.config, config_file, config_from=self, **kwargs)
        else:
            config_model = settings.get_config(
                config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        config_model.check(
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Check custom model calculator
        if model_calculator is not None:
            
            # Assign model calculator
            self.model_calculator = model_calculator
            self.model_restart = True
        
            # Add calculator info to configuration dictionary
            if hasattr(model_calculator, "get_info"):
                config_model.update(
                    model_calculator.get_info())
        
        else:
        
            # Get model calculator
            model_calculator, restart = self._get_model_calculator(
                config=config,
                config_file=config_file,
                model_type=model_type,
                model_checkpoint=model_checkpoint,
                )
            
            # Assign model calculator
            self.model_calculator = model_calculator
            self.model_restart = restart

        return

    def get_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ) -> torch.nn.Module:
        """
        Return calculator model class object

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: int, optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """
        
        # Get model calculator
        model_calculator, restart = self._get_model_calculator(
            config=config,
            config_file=config_file,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            **kwargs,
            )

        return model_calculator

    def _get_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'last',
        **kwargs,
    ) -> (torch.nn.Module, bool):
        """
        Return calculator model class object and restart flag.

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: int, optional, default 'last'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object
        bool
            Restart flag, True if checkpoint file is loaded.

        """

        ########################################
        # # # Check Model Calculator Input # # #
        ########################################

        # Assign model parameter configuration library
        if config is None:
            config_model = settings.get_config(
                self.config, config_file, config_from=self, **kwargs)
        else:
            config_model = settings.get_config(
                config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        config_model.check(
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Check requested model type
        if model_type is None and config_model.get('model_type') is None:
            model_type = settings._default_calculator_model
        elif model_type is None:
            model_type = config_model['model_type']

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign model calculator
        model_calculator = model.get_model_calculator(
            model_type,
            config=config_model,
            **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(model_calculator, "get_info"):
            config_model.update(
                model_calculator.get_info())

        # Initialize checkpoint file manager and load best model
        filemanager = model.FileManager(config_model, **kwargs)
        
        # If checkpoint is None or 'best', load best model
        if (
            model_checkpoint is None
            or utils.is_string(model_checkpoint) 
            and model_checkpoint.lower() == 'best'
        ):
            checkpoint = filemanager.load_checkpoint(best=True)
        # Else if checkpoint is 'last', load last checkpoint file
        elif (
            utils.is_string(model_checkpoint) 
            and model_checkpoint.lower() == 'last'
        ):
            checkpoint = filemanager.load_checkpoint(
                best=False,
                num_checkpoint=None)
        # Else if checkpoint is integer, load checkpoint file of respective
        # epoch number 
        elif utils.is_integer(model_checkpoint):
            checkpoint = filemanager.load_checkpoint(
                best=False,
                num_checkpoint=model_checkpoint)
        else:
            raise ValueError(
                "Input for 'model_checkpoint' must be either None to load "
                + "the best model checkpoint or "
                + "an integer of a respective checkpoint file.")
        if checkpoint is None:
            logger.info(f"INFO:\nNo checkpoint file loaded.\n")
            restart = False
        else:
            model_calculator.load_state_dict(
                checkpoint['model_state_dict'])
            logger.info(f"INFO:\nCheckpoint file loaded.\n")
            restart = True

        return model_calculator, restart

    def train(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize and start model training

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to config json file (str)

        """

        ################################
        # # # Check Training Input # # #
        ################################

        # Assign model parameter configuration library
        if config is None:
            config_train = settings.get_config(
                self.config, config_file, config_from=self, **kwargs)
        else:
            config_train = settings.get_config(
                config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        config_train.check(
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        #################################
        # # # Assign Reference Data # # #
        #################################

        # Assign DataContainer
        if self.data_container is None:

            self.data_container = self.get_DataContainer(
                config=config_train,
                **kwargs)

        # Add data container info to configuration dictionary
        if hasattr(self.data_container, "get_info"):
            config_train.update(self.data_container.get_info())

        ####################################
        # # # Prepare Model Calculator # # #
        ####################################

        # Assign model calculator to train
        if self.model_calculator is None:

            # Initialize model calculator
            model_calculator, restart = self._get_model_calculator(
                config=config_train,
                model_checkpoint='last',
                **kwargs)

            # Get property scaling guess from reference data if model did
            # not load from checkpoint.
            if not restart:

                # Get property statistics
                property_stats = self.data_container.get_property_scaling(
                    overwrite=False)
                
                # Assign property statistics as initial property scaling
                # parameters.
                model_calculator.set_property_scaling(
                    scaling_parameter=property_stats)

        else:
            
            model_calculator = self.model_calculator

        # Add calculator info to configuration dictionary
        if hasattr(model_calculator, "get_info"):
            config_train.update(
                model_calculator.get_info())

        ###############################
        # # # Prepare NNP Trainer # # #
        ###############################

        # Assign model calculator trainer
        if self.trainer is None:

            self.trainer = train.Trainer(
                config=config_train,
                data_container=self.data_container,
                model_calculator=self.model_calculator,
                **kwargs
                )
        exit()
        # Start training
        self.trainer.train()

        return
