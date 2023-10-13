import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import optimize

from .. import data
from .. import settings
from .. import utils
from .. import interface

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
        sample_systems_optimize: Optional[bool] = None,
        sample_systems_optimize_fmax: Optional[float] = None,
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
        sample_systems_optimize: bool, optional, default False
            Instruction flag if the system coordinates shall be
            optimized using the ASE calculator defined by 'sample_calculator'.
        sample_systems_optimize_fmax: float, optional, default 0.01
            Instruction flag, if the system coordinates shall be
            optimized using the ASE calculator defined by 'sample_calculator'.

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
        config_update = {}
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
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        config.update(config_update)

        # Check system input
        if self.sample_systems is None:
            logger.warning(
                "WARNING:\nNo input in 'sample_systems' is given!\n"
                + "Please provide either a chemical structure file or "
                + "an ASE Atoms object as initial sample structure.")
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
                "Sample system input 'sample_systems' and "
                + "'sample_systems_format' have different input size of "
                + f"{len(self.sample_systems):d} and "
                + f"{len(self.sample_systems_format):d}, respectively.")

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
        sample_systems_atoms: Optional[Union[object, List[object]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Assign calculator to a list of sample ASE Atoms objects
        """

        # Check input
        if sample_systems_atoms is None:
            sample_systems_atoms = self.sample_systems_atoms
        elif not utils.is_array_like(sample_systems_atoms):
            sample_systems_atoms = [sample_systems_atoms]
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
                    f"Requested property '{prop:s}' is not implemented "
                    + f"in the ASE calculator '{sample_calculator}'! "
                    + "Available ASE calculator properties are:\n"
                    + f"{sample_calculator.implemented_properties}")

        # Define positions and property units
        self.sample_unit_positions = 'Ang'
        self.sample_unit_properties = {
            prop: interface.ase_calculator_units.get(prop)
            for prop in self.sample_properties}

    def get_info(self):
        """
        Dummy function for sampling parameter dictionary
        """
        return {}

    def run(
        self,
        sample_systems_idx: Optional[Union[int, List[int]]] = None,
    ):
        """
        Perform sampling of all sample systems or a selection of them.

        Parameters
        ----------

        sample_systems_idx: (int, list(int)), optional, default None
            Index or list of indices to run MD sampling only
            for the respective systems of the sample system list
        """

        ################################
        # # # Check Sampling Input # # #
        ################################

        # Collect sampling parameters
        config_sample = {
            f'{self.sample_counter}_{self.sample_tag}':
                self.get_info()
            }

        # Check sample system selection
        sample_systems_selection = self.check_systems_idx(sample_systems_idx)

        # Update sampling parameters
        config_sample['sample_systems_idx'] = sample_systems_idx

        # Update configuration file with sampling parameters
        if 'sampler_schedule' not in self.config:
            self.config['sampler_schedule'] = {}
        self.config['sampler_schedule'].update(config_sample)

        # Increment sample counter
        self.config['sample_counter'] = self.sample_counter
        self.sample_counter += 1

        ###############################
        # # # Perform MD Sampling # # #
        ###############################

        # Iterate over systems
        for isys, system in enumerate(self.sample_systems_atoms):

            # Skip unselected system samples
            if not sample_systems_selection[isys]:
                continue

            # If requested, perform structure optimization
            if self.sample_systems_optimize:

                # Assign ASE optimizer
                ase_optimizer = optimize.BFGS

                # Perform structure optimization
                ase_optimizer(system).run(
                    fmax=self.sample_systems_optimize_fmax)

            # Start normal mode sampling
            self.run_system(system)

    def run_system(self, system):
        raise NotImplementedError()

    def get_properties(self, system):
        """
        Collect system properties and calculator results
        """

        return interface.get_ase_properties(system, self.sample_properties)


    def check_systems_idx(self, systems_idx):
        """
        Check sample system selection input
        """
    
        if systems_idx is None:
            
            # Select all
            systems_selection = np.ones(
                len(self.sample_systems_atoms), dtype=bool)
            
        elif utils.is_integer(systems_idx):
            
            # Check selection index and sample system length
            if (
                    systems_idx >= len(self.sample_systems_atoms)
                    or systems_idx < -len(self.sample_systems_atoms)
                ):
                raise ValueError(
                    f"Sample systems selection 'systems_idx' " +
                    f"({systems_idx}) " +
                    f"is out of range of loaded system samples" +
                    f"({len(self.sample_systems_atoms)})!")
            
            # Select system of index
            systems_selection = np.zeros(
                len(self.sample_systems_atoms), dtype=bool)
            systems_selection[systems_idx] = True
            
        elif utils.is_integer_array(systems_idx):
            
            # Select systems of respective indices
            systems_selection = np.zeros(
                len(self.sample_systems_atoms), dtype=bool)
            for idx in systems_idx:
                
                # Check selection index and sample system length
                if (
                        idx >= len(self.sample_systems_atoms)
                        or idx < -len(self.sample_systems_atoms)
                    ):
                    raise ValueError(
                        f"Sample systems selection in 'systems_idx' " +
                        f"({idx}) " +
                        f"is out of range of loaded system samples" +
                        f"({len(self.sample_systems_atoms)})!")
                
                # Select system
                systems_selection[idx] = True
                
        else:
            
            raise ValueError(
                f"Sample systems selection in 'systems_idx' " +
                f"has a wrong type '{type(systems_idx)}'!\n"
                )
        
        return systems_selection
