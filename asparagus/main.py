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
from .src import interface

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
            'train', 'ase', 'pycharmm', ... (Still defined?)
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
        # Model calculator
        self.model_calculator = None
        # Model trainer
        self.trainer = None
        # Model testing
        self.tester = None

        return

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

    def train(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize and start model training
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
        if self.model_calculator is None:

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
            self.model_calculator = self._get_Calculator(
                config_train,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_train.update(
                self.model_calculator.get_info(),
                verbose=False)

        ###############################
        # # # Prepare NNP Trainer # # #
        ###############################

        # Assign NNP Trainer
        if self.trainer is None:

            self.trainer = train.Trainer(
                config_train,
                self.data_container,
                self.model_calculator,
                **kwargs
                )

        # Start training
        self.trainer.train()

        return

    def test(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        test_datasets: Optional[Union[str, List[str]]] = None,
        test_directory: Optional[str] = None,
        test_checkpoint: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize and start model testing

        Parameters
        ----------

        test_datasets: (str, list(str)) optional, default ['test']
            A string or list of strings to define the data sets ('train', 
            'valid', 'test') of which the evaluation will be performed.
            By default it is just the test set of the data container object.
            Inputs 'full' or 'all' requests the evaluation of all sets.
        test_directory: str, optional, default '.'
            Directory to store evaluation graphics and data.
        test_checkpoint: int, optional, default None
            If None, load best model checkpoint. Otherwise define a checkpoint
            index number of the respective checkpoint file.
        """
        
        ################################
        # # # Check Training Input # # #
        ################################

        # Assign model parameter configuration library
        if config is None:
            config_test = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_test = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_test.check()

        ##################################
        # # # Prepare Reference Data # # #
        ##################################

        # Assign DataContainer
        if self.data_container is None:

            self.data_container = self._get_DataContainer(
                config_test,
                **kwargs)

        # Add data container info to configuration dictionary
        if hasattr(self.data_container, "get_info"):
            config_test.update(
                self.data_container.get_info(),
                verbose=False)

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign NNP calculator
        if self.model_calculator is None:

            # Assign NNP calculator model
            self.model_calculator = self._get_Calculator(
                config_test,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_test.update(
                self.model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_test, **kwargs)
        if test_checkpoint is None:
            latest_checkpoint = filemanager.load_checkpoint(best=True)
        elif utils.is_integer(test_checkpoint):
            latest_checkpoint = filemanager.load_checkpoint(
                num_checkpoint=test_checkpoint)
        else:
            raise ValueError(
                "Input 'test_checkpoint' must be either None to load best "
                + "model checkpoint or an integer of a respective checkpoint "
                + "file.")
        self.model_calculator.load_state_dict(
            latest_checkpoint['model_state_dict'])
        
        ######################################
        # # # Prepare and Run NNP Tester # # #
        ######################################
        
        # Assign model prediction tester
        self.tester = train.Tester(
            config_test,
            self.data_container,
            test_datasets=test_datasets)
        
        # Evaluation of the test set
        self.tester.test(
            self.model_calculator, 
            test_directory=test_directory,
            **kwargs)

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


    def get_ase_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[int] = None,
        atoms: Optional[Union[object, List[object]]] = None,
        atoms_charge: Optional[Union[float, List[float]]] = None,
        implemented_properties: Optional[List[str]] = None,
        use_neighbor_list: Optional[bool] = None,
        label: Optional[str] = 'asparagus',
        **kwargs,
    ) -> Callable:
        """
        initialize ASE calculator class object of the model calculator
        """

        ######################################
        # # # Check ASE Calculator Input # # #
        ######################################

        # Assign model parameter configuration library
        if config is None:
            config_ase = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_ase = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_ase.check()

        # Check for empty config dictionary
        if "model_directory" not in config_ase:
            raise SyntaxError(
                "Configuration does not provide information for a model "
                + "calculator. Please check the input in 'config'.")

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign NNP calculator
        if self.model_calculator is None:

            # Assign NNP calculator model
            self.model_calculator = self._get_Calculator(
                config_ase,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_ase.update(
                self.model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_ase, **kwargs)
        if model_checkpoint is None:
            latest_checkpoint = filemanager.load_checkpoint(best=True)
        elif utils.is_integer(model_checkpoint):
            latest_checkpoint = filemanager.load_checkpoint(
                num_checkpoint=model_checkpoint)
        else:
            raise ValueError(
                "Input 'model_checkpoint' must be either None to load best "
                + "model checkpoint or an integer of a respective checkpoint "
                + "file.")
        self.model_calculator.load_state_dict(
            latest_checkpoint['model_state_dict'])

        ##################################
        # # # Prepare ASE Calculator # # #
        ##################################

        self.ase_calculator = interface.ASE_Calculator(
            self.model_calculator,
            atoms=atoms,
            atoms_charge=atoms_charge,
            implemented_properties=implemented_properties,
            use_neighbor_list=use_neighbor_list,
            label=label,
            )

        return self.ase_calculator

    def get_pycharmm_calculator(self,
        # Total number of atoms
        num_atoms: int,
        # PhysNet atom indices
        ml_atom_indices: List[int],
        # PhysNet atom numbers
        ml_atom_numbers: List[int],
        # Fluctuating ML charges for ML-MM electrostatic interaction
        ml_fluctuating_charges: bool,
        # System atom charges (All atoms)
        ml_mm_atoms_charge: List[float],
        # Total charge of the system
        ml_total_charge: Optional[float],
        # Cutoff distance for ML/MM electrostatic interactions
        mlmm_rcut: float,
        # Cutoff width for ML/MM electrostatic interactions
        mlmm_width: float,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[int] = None,**kwargs):

        """
        Initialize PyCharmm calculator class object of the model calculator
        """


        ###########################################
        # # # Check PyCharmm Calculator Input # # #
        ###########################################

        # Assign model parameter configuration library
        if config is None:
            config_pycharmm = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_pycharmm = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_pycharmm.check()

        # Check for empty config dictionary
        if "model_directory" not in config_pycharmm:
            raise SyntaxError(
                "Configuration does not provide information for a model "
                + "calculator. Please check the input in 'config'.")

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign NNP calculator
        if self.model_calculator is None:
            # Assign NNP calculator model
            self.model_calculator = self._get_Calculator(
                config_pycharmm,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_pycharmm.update(
                self.model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_pycharmm, **kwargs)
        if model_checkpoint is None:
            latest_checkpoint = filemanager.load_checkpoint(best=True)
        elif utils.is_integer(model_checkpoint):
            latest_checkpoint = filemanager.load_checkpoint(
                num_checkpoint=model_checkpoint)
        else:
            raise ValueError(
                "Input 'model_checkpoint' must be either None to load best "
                + "model checkpoint or an integer of a respective checkpoint "
                + "file.")
        self.model_calculator.load_state_dict(
            latest_checkpoint['model_state_dict'])

        if ml_total_charge is None:
            ml_total_charge = 0

        ##################################
        # # # Prepare Calculator # # #
        ##################################

        self.pycharmm_calculator = interface.PyCharmm_Calculator(
            self.model_calculator,
            num_atoms=num_atoms,
            ml_atom_indices=ml_atom_indices,
            ml_atom_numbers=ml_atom_numbers,
            ml_fluctuating_charges=ml_fluctuating_charges,
            ml_mm_atoms_charge=ml_mm_atoms_charge,
            ml_total_charge=ml_total_charge,
            mlmm_rcut=mlmm_rcut,
            mlmm_width=mlmm_width,
            )

        return self.pycharmm_calculator

