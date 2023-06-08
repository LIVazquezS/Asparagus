import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import itertools

import ase
from ase import optimize
from ase import vibrations
from ase import units
from ase.visualize import view

from .. import data
from .. import model
from .. import settings
from .. import utils
from .. import interface
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Sampler']


# ======================================
# General Conformation Sampler Class
# ======================================

class Sampler:
    """
    Conformation Sampler main class for generation of reference structures.
    """
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        sample_directory: Optional[str] = None,
        sample_data_file: Optional[str] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        sample_properties: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Sampler class.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        sample_directory: str, optional, default ''
            Working directory where to store the database file of the sampled
            system data, eventually temporary ASE calculator files,
            ASE trajectory files and/or model calculator files.
        sample_data_file: str, optional, default 'sample.db'
            Database file name to store a selected set of systems with
            computed reference data.
        sample_systems: (str, list, object), optional, default ''
            System coordinate file or a list of system coordinate files or
            ASE atoms objects that are considered as initial conformations for
            reference structure sampling.
        sample_systems_format: (str, list), optional, default ''
            System coordinate file format string (e.g. 'xyz') for the
            definition in 'sample_systems' in case of file paths.
        sample_calculator: (str, callable object), optional, default 'XTB'
            Definition of the ASE calculator type for reference data
            computation. The input can be either directly a ASE calculator
            class object or a string with available ASE calculator classes.
        sample_calculator_args: dict, optional, default {}
            In case of string type input for 'sample_calculator', this
            dictionary is passed as keyword arguments at the initialization
            of the ASE calculator.
        sample_properties: List[str], optional, default None
            List of system properties which are computed by the ASE
            calculator class. Requested properties will be checked with the
            calculator available property list and return an error when one
            requested property is unavailable. By default all available
            properties will be stored.
        
        Returns
        -------
        callable object
            Sampler class object
        """
        
        #####################################
        # # # Check Sampler Class Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(config)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        #config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'kwargs', '__class__']:
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
            #config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        #config.update(config_update)
        
        # Check system input
        if self.sample_systems is None:
            logger.warning(
                f"WARNING:\nNo input in 'sample_systems' is given!\n" +
                f"Please provide either a chemical structure file or " +
                f"an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

        # Initialize sampling counter
        if config.get('sample_counter') is None:
            self.sample_counter = 1
        else:
            self.sample_counter = config.get('sample_counter') + 1

        # Set global configuration as class parameter
        self.config = config
        
        # Generate working directory
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

        ###########################
        # # # Prepare Systems # # #
        ###########################
        
        self.sample_systems_atoms = self.read_systems()

        #####################################
        # # # Prepare Sample Calculator # # #
        #####################################
        
        self.assign_calculator()



    def read_systems(self):
        """
        Read sample system files and return list of respective 
        ASE atoms objects
        """
        
        # Prepare system structure input by converting to matching lists
        if utils.is_string(self.sample_systems):
            self.sample_systems = [self.sample_systems]
        elif utils.is_ase_atoms(self.sample_systems):
            self.sample_systems = [self.sample_systems]
        if self.sample_systems_format is None:
            self.sample_systems_format = [None]*len(self.sample_systems)
        elif utils.is_string(self.sample_systems_format):
            self.sample_systems_format = (
                [self.sample_systems_format]*len(self.sample_systems))
        elif len(self.sample_systems) != len(self.sample_systems_format):
            raise ValueError(
                f"Sample system input 'sample_systems' and " +
                f"'sample_systems_format' have different input size of " +
                f"{len(self.sample_systems):d} and " +
                f"{len(self.sample_systems_format):d}, respectively.")
        
        # Iterate over system input and eventually read file to store as
        # ASE Atoms object
        sample_systems_atoms = []
        for system, system_format in zip(
                self.sample_systems, self.sample_systems_format):

            # Check for ASE Atoms object or read system file
            if utils.is_ase_atoms(system):
                sample_systems_atoms.append(system)
            else:
                sample_systems_atoms.append(
                    ase.io.read(system, format=system_format))
                
        return sample_systems_atoms


    def assign_calculator(
        self,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        ):
        """
        Assign calculator to a list of sample ASE Atoms objects
        """
        
        # Check input
        if sample_calculator is None:
            sample_calculator = self.sample_calculator
        if sample_calculator_args is None:
            sample_calculator_args = self.sample_calculator_args
        
        # Get ASE calculator
        sample_calculator, sample_calculator_tag = (
            interface.get_ase_calculator(
                sample_calculator,
                sample_calculator_args)
            )
        
        # Store calculator tag name
        self.sample_calculator_tag = sample_calculator_tag
        
        # Assign ASE calculator
        self.sample_calculator = sample_calculator
        for system in self.sample_systems_atoms:
            system.set_calculator(sample_calculator)

        # Check requested system properties
        for prop in self.sample_properties:
            # TODO Special calculator properties list for special properties
            # not supported by ASE such as, e.g., charge, hessian, etc.
            if prop not in sample_calculator.implemented_properties:
                raise ValueError(
                    f"Requested property '{prop:s}' is not implemented " +
                    f"in the ASE calculator '{sample_calculator}'! " +
                    f"Available ASE calculator properties are:\n" +
                    f"{sample_calculator.implemented_properties}")

        # Define positions and property units
        self.sample_unit_positions = 'Ang'
        self.sample_unit_properties = {
            prop: interface.ase_calculator_units.get(prop)
            for prop in self.sample_properties}
        
    
