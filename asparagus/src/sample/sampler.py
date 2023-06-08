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
    Conformation Sampler class for generation of reference structures.
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
        
        # Set global configuration as class parameter
        self.config = config
        
        # Check system input
        if self.sample_systems is None:
            logger.warning(
                f"WARNING:\nNo input in 'sample_systems' is given!\n" +
                f"Please provide either a chemical structure file or " +
                f"an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

        # Initialize sampling counter
        self.sample_counter = 0
        
        ###########################
        # # # Prepare Systems # # #
        ###########################

        # Generate working directory
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

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
        self.sample_systems_atoms = []
        for system, system_format in zip(
                self.sample_systems, self.sample_systems_format):

            # Check for ASE Atoms object or read system file
            if utils.is_ase_atoms(system):
                self.sample_systems_atoms.append(system)
            else:
                self.sample_systems_atoms.append(
                    ase.io.read(system, format=system_format))

        #####################################
        # # # Prepare Sample Calculator # # #
        #####################################

        # Get ASE calculator
        self.sample_calculator = interface.get_ase_calculator(
            self.sample_calculator,
            self.sample_calculator_args)

        # Assign ASE calculator
        for system in self.sample_systems_atoms:
            system.set_calculator(self.sample_calculator)

        # Check requested system properties
        for prop in self.sample_properties:
            if prop not in self.sample_calculator.implemented_properties:
                raise ValueError(
                    f"Requested property '{prop:s}' is not implemented " +
                    f"in the ASE calculator '{self.sample_calculator:s}'! " +
                    f"Available ASE calculator properties are:\n" +
                    f"{self.sample_calculator.implemented_properties}")

        # Define positions and property units
        self.sample_unit_positions = 'Ang'
        self.sample_unit_properties = {
            prop: interface.ase_calculator_units.get(prop)
            for prop in self.sample_properties}
        

        
    def normal_mode_sampling(
        self,
        nms_data_file: Optional[str] = None,
        nms_systems_optimize: Optional[bool] = True,
        nms_systems_optimize_fmax: Optional[float] = 0.01,
        nms_harmonic_energy_step: Optional[float] = None,
        nms_energy_limits: Optional[Union[float, List[float]]] = None,
        nms_limit_of_coupling: Optional[int] = None,
        nms_limit_of_steps: Optional[int] = None,
        **kwargs,
    ):
        """
        Perform Normal Mode Sampling

        Parameters
        ----------

        nms_data_file: str, optional, default 'sample.db'
            Database file name to store the sampled systems with computed
            reference data.
        nms_systems_optimize: bool, optional, default False
            Instruction flag, if the system coordinates shall be
            optimized using the ASE calculator defined by 'sample_calculator'.
        nms_systems_optimize_fmax: float, optional, default 0.001
            Instruction flag, if the system coordinates shall be
            optimized using the ASE calculator defined by 'sample_calculator'.
        nms_harmonic_energy_step: float, optional, default 0.05
            Within the harmonic approximation the initial normal mode
            displacement from the equilibrium positions is scaled to match
            the potential difference with the given energy value.
        nms_energy_limits: (float, list(float)), optional, default 1.0
            Potential energy limit in eV from the initial system conformation 
            to which additional normal mode displacements steps are added.
            If one numeric value is give, the energy limit is used as upper 
            potential energy limit and the lower limit in case the initial
            system conformation might not be the global minimum.
            If a list with two numeric values are given, the first two are the lower and upper potential energy limit, respectively.
        nms_limit_of_coupling: int, optional, default 2
            Maximum limit of coupled normal mode displacements to sample
            the system conformations.
        nms_limit_of_steps: int, optional, default 10
            Maximum limit of coupled normal mode displacements to sample
            the system conformations.
        """
        
        ################################
        # # # Initialize NMS Class # # #
        ################################
        
        # Check sample data file
        if nms_data_file is None:
            nms_data_file = os.path.join(
                self.sample_directory, f'{self.sample_counter:d}_nms.db')
        elif not utils.is_string(nms_data_file):
            raise ValueError(
                f"Sample data file 'nms_data_file' must be a string " +
                f"of a valid file path but is of type " + f"'{type(nms_data_file)}'.")
        
        # Initialize Normal Mode Sampler class object
        self.normal_mode_sampler = sample.NormalModeSampler(
            nms_harmonic_energy_step=nms_harmonic_energy_step,
            nms_energy_limits=nms_energy_limits,
            nms_limit_of_coupling=nms_limit_of_coupling,
            nms_limit_of_steps=nms_limit_of_steps,
            **kwargs
            )
        
        # Update configuration file with sampling parameters
        sample_tag = 'nms'
        config_nms = {
            f'{self.sample_counter}_{sample_tag}': 
                self.normal_mode_sampler.get_info()
            }
        self.config.update(config_nms)
        
        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################
        
        self.sample_properties += ['charge']
        self.sample_unit_properties.update({'charge': 'e'})
        nms_dataset = data.DataSet(
            nms_data_file,
            self.sample_unit_positions,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=True)
        
        ########################################
        # # # Perform Normal Mode Sampling # # #
        ########################################

        # Iterate over systems
        for system in self.sample_systems_atoms:

            # If requested, perform structure optimization
            if nms_systems_optimize:

                # Assign ASE optimizer
                ase_optimizer = optimize.BFGS

                # Perform structure optimization
                ase_optimizer(system).run(
                    fmax=nms_systems_optimize_fmax)
                
            # Start normal mode sampling
            self.normal_mode_sampler.run(
                system,
                nms_dataset,
                self.sample_properties,
                self.sample_directory)

