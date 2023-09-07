import os
import sys
import logging
import datetime
from typing import Optional, List, Dict, Tuple, Union, Callable

import numpy as np 

import torch
#import pytorch_lightning as pl
#from pytorch_lightning.accelerators import GPUAccelerator #TODO

from .src import settings
from .src import utils
from .src import data
from .src import model
from .src import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Asparagus']


class Asparagus(torch.nn.Module):
    """
    Neural network potential (NNP) main class to check and parse the tasks and
    model properties.
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):
        """
        Neural Network Potential

        Parameters
        ----------

        job: str
            Define the kind of job to be performed by the NNP, e.g.,
            'train', 'ase', 'pycharmm', ...
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
                Main PhysNetSuite object to direct tasks.
        """

        super().__init__()

        #############################
        # # # Check Model Input # # #
        #############################

        # Initialize model parameter configuration dictionary
        # Keyword arguments overwrite entries in the configuration dictionary
        self.config = settings.get_config(
            config, config_file, **kwargs)

        # Check model parameter configuration and set default
        self.config.check()

        # Set global parameters
        if self.config.get("model_dtype") is not None:
            settings.set_global_dtype(self.config.get("model_dtype"))
        if self.config.get("model_device") is not None:
            settings.set_global_device(self.config.get("model_device"))

        ###########################
        # # # Class Parameter # # #
        ###########################

        # DataContainer of reference data
        self.data_container = None
        # NN Calculator
        self.calculator = None
        # NN trainer
        self.trainer = None
        # NN testing
        self.tester = None

        return


    def train(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize and start NN training
        """

        ################################
        # # # Check Training Input # # #
        ################################

        # Assign model parameter configuration library
        if config is None:
            config_train = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_train = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_train.check()

        ##################################
        # # # Prepare Reference Data # # #
        ##################################

        # Assign DataContainer
        if self.data_container is None:

            self.data_container = self._get_DataContainer(
                config_train,
                **kwargs)

            # Assign training, validation and test data loader
            self.data_train = self.data_container.train_loader
            self.data_valid = self.data_container.valid_loader
            self.data_test = self.data_container.test_loader

        # Add data container info to configuration dictionary
        if hasattr(self.data_container, "get_info"):
            config_train.update(
                self.data_container.get_info(),
                verbose=False)

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Set global dropout rate value
        if config_train.get("trainer_dropout_rate") is not None:
            settings.set_global_rate(config_train.get("trainer_dropout_rate"))

        # Assign NNP calculator
        if self.calculator is None:

            # Get property scaling guess from reference data to link
            # 'normalized' output to average reference data shift and
            # distribution width.
            if ('model_properties_scaling' not in kwargs.keys() or
                    'model_properties_scaling' not in config_train):

                model_properties_scaling = (
                    self.data_container.get_property_scaling())

                config_train.update(
                    {'model_properties_scaling': model_properties_scaling})

            # Assign NNP calculator model
            self.calculator = self._get_Calculator(
                config_train,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.calculator, "get_info"):
            config_train.update(
                self.calculator.get_info(),
                verbose=False)

        ## Example run of calculator
        #for batch in self.data_train:

            #result = self.calculator(
                #batch['atoms_number'], 
                #batch['atomic_numbers'], 
                #batch['positions'], 
                #batch['idx_i'], 
                #batch['idx_j'],
                #batch['charge'],
                #batch['atoms_seg'])
            #print(result)

        ###############################
        # # # Prepare NNP Trainer # # #
        ###############################

        # Assign NNP Trainer
        if self.trainer is None:

            self.trainer = train.Trainer(
                config_train,
                self.data_container,
                self.calculator,
                **kwargs
                )

        # Start training
        self.trainer.train()

        return

    def test_model(
        self, 
        checkpoint=None, verbose=True,
        save_npz=False, npz_name='test_vals.npz',
        save_csv=False, csv_name='test_vals.csv',
        plot=False, plots_to_show=None, save_plots=False, show_plots=False,
        residual_plots=False, residuals_to_show=None, save_residuals=False, show_residuals=False,
        histogram_plots=False, histograms_to_show=None, save_histograms=False, show_histograms=False,
        **kwargs):
        """
        Testing model in the test set.
        
        I am not sure if checking the config file is required here.

        """
        ##################################
        # # # Prepare Reference Data # # #
        ##################################

        # Assign DataContainer
        if self.data_container is None:
            self.data_container = self._get_DataContainer(
                self.config,
                **kwargs)

        # Check for checkpoint
        if checkpoint is not None:
            self.checkpoint = checkpoint

        # Assign NNP calculator
        if self.calculator is None:

            # Get property scaling guess from reference data to link
            # 'normalized' output to average reference data shift and
            # distribution width.
            if ('model_properties_scaling' not in kwargs.keys() or
                    'model_properties_scaling' not in self.config):
                model_properties_scaling = (
                        self.data_container.get_property_scaling())

                self.config.update(
                        {'model_properties_scaling': model_properties_scaling})

            # Assign NNP calculator model
            self.calculator = self._get_Calculator(self.config)

        # Assign NNP Tester
        if self.tester is None:
            self.tester = train.Testing(
                self.config, 
                self.data_container,
                self.calculator, 
                self.checkpoint)
        
        # Run testing
        self.tester.test(verbose=verbose,save_npz=save_npz, npz_name=npz_name,
             save_csv=save_csv, csv_name=csv_name,
             plot=plot, plots_to_show=plots_to_show, save_plots=save_plots, show_plots=show_plots,
             residual_plots=residual_plots, residuals_to_show=residuals_to_show, save_residuals=save_residuals,show_residuals=show_residuals,
             histogram_plots=histogram_plots, histograms_to_show=histograms_to_show, save_histograms=save_histograms, show_histograms=show_histograms)
        
        return

    def _get_DataContainer(
        self,
        config: Union[dict, object],
        **kwargs,
    ) -> Callable:
        """
        Initialize DataContainer object
        """

        # Check for DataContainer class in input
        if kwargs.get('data_container') is not None:

            return kwargs.get('data_container')

        elif config.get('data_container') is not None:

            return config.get('data_container')

        # Otherwise initialize DataContainer class
        else:

            return data.DataContainer(
                config=config,
                **kwargs)


    def _get_Calculator(
        self,
        config: Union[dict, object],
        **kwargs,
    ) -> Callable:
        """
        Initialize NNP calculator object
        """

        # Check for NNP calculator class in input
        if kwargs.get('model_calculator') is not None:

            return kwargs.get('model_calculator')

        elif config.get('model_calculator') is not None:

            return config.get('model_calculator')

        # Otherwise initialize NNP calculator class
        else:

            return model.get_calculator(
                config=config,
                **kwargs)


    def get_directory_generic(
        self,
        directory_tag: Optional[str] = '',
    ) -> str:

        # Check directory label tag
        if not utils.is_string(directory_tag):
            raise SyntaxError(
                f"Model directory label 'directory_tag' is not " +
                f"of type string but of type '{type(directory_tag)}'!")

        # Get generic label
        label = (
            directory_tag + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

        return label
