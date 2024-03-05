import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import optimize

from .. import sample

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


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    sample_data_file: str, optional, default None
        Database file name to store a selected set of systems with
        computed reference data. If None, data file name is the respective
        sample method tag.
    sample_data_file_format: str, optional, default None
        Database file format. If None, data file prefix is taken as file
        format tag.
    sample_directory: str, optional, default None
        Working directory where to store eventually temporary ASE
        calculator files, ASE trajectory files and/or model calculator
        files. If None, files will be stored in parent directory.
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
    sample_data_overwrite: bool, optional, default False
        If False, add new sampling data to an eventually existing data
        file. If True, overwrite an existing one.
    sample_tag: str, optional, default 'sample'
        Sampling method tag of the specific sampling methods for
        log and ASE trajectory files or the data file name if not defined.

    Returns
    -------
    callable object
        Sampler class object
    """

    # Default arguments for sample module
    _default_args = {
        'sample_directory':             None,
        'sample_data_file':             None,
        'sample_data_file_format':      None,
        'sample_systems':               None,
        'sample_systems_format':        None,
        'sample_calculator':            'XTB',
        'sample_calculator_args':       {},
        'sample_properties':            ['energy', 'forces', 'dipole'],
        'sample_systems_optimize':      False,
        'sample_systems_optimize_fmax': 0.001,
        'sample_data_overwrite':        False,
        'sample_tag':                   'sample',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'sample_directory':             [utils.is_string, utils.is_None],
        'sample_data_file':             [utils.is_string, utils.is_None],
        'sample_data_file_format':      [utils.is_string, utils.is_None],
        'sample_systems':               [utils.is_string,
                                        utils.is_string_array,
                                        utils.is_ase_atoms,
                                        utils.is_ase_atoms_array],
        'sample_systems_format':        [utils.is_string, 
                                         utils.is_string_array],
        'sample_calculator':            [utils.is_string, 
                                         utils.is_object],
        'sample_calculator_args':       [utils.is_dictionary],
        'sample_properties':            [utils.is_string, 
                                         utils.is_string_array],
        'sample_systems_optimize':      [utils.is_bool, 
                                         utils.is_boolean_array],
        'sample_systems_optimize_fmax': [utils.is_numeric],
        'sample_data_overwrite':        [utils.is_bool],
        'sample_tag':                   [utils.is_string],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        sample_data_file: Optional[str] = None,
        sample_data_file_format: Optional[str] = None,
        sample_directory: Optional[str] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        sample_properties: Optional[List[str]] = None,
        sample_systems_optimize: Optional[bool] = None,
        sample_systems_optimize_fmax: Optional[float] = None,
        sample_data_overwrite: Optional[bool] = None,
        sample_tag: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Sampler class.
        """

        #####################################
        # # # Check Sampler Class Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=locals().items(),
            argsskip=['self', 'config', 'metadata', 'kwargs', '__class__'],
            check_default=utils.get_default_args(self, sample),
            check_dtype=utils.get_dtype_args(self, sample)
        )

        # Set global configuration as class parameter
        self.config = config

        # Check system input
        if self.sample_systems is None:
            logger.warning(
                "WARNING:\nNo input in 'sample_systems' is given!\n"
                + "Please provide either a chemical structure file or "
                + "an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

        ############################
        # # # Prepare Sampling # # #
        ############################

        # Initialize sampling counter
        if config.get('sample_counter') is None:
            self.sample_counter = 1
        else:
            self.sample_counter = config.get('sample_counter') + 1

        # Generate working directory
        if self.sample_directory is None or not len(self.sample_directory):
            self.sample_directory = '.'
        elif not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

        # Check sample data file
        if self.sample_data_file is None:
            self.sample_data_file = f'{self.sample_tag:s}.db'
        elif not utils.is_string(self.sample_data_file):
            raise ValueError(
                "Sample data file 'sample_data_file' must be a string "
                + "of a valid file path but is of type "
                + f"'{type(self.sample_data_file)}'.")
        self.sample_data_file_format = sample_data_file_format

        #self.sample_systems_atoms = self.read_systems()

        #####################################
        # # # Prepare Sample Calculator # # #
        #####################################

        # Get ASE calculator
        self.sample_calculator, self.sample_calculator_tag = (
            interface.get_ase_calculator(
                self.sample_calculator,
                self.sample_calculator_args)
            )

        # Check requested system properties
        self.check_properties()

        # Store calculator tag name
        self.sample_calculator_tag = self.sample_calculator_tag

        #############################
        # # # Prepare Optimizer # # #
        #############################

        if self.sample_systems_optimize:

            # Assign ASE optimizer
            optimizer_tag = "bfgs"
            self.ase_optimizer = optimize.BFGS

            # Assign optimization log and  trajectory file name
            self.ase_optimizer_log_file = os.path.join(
                self.sample_directory,
                f'{self.sample_counter:d}_{optimizer_tag:s}.log')
            self.ase_optimizer_trajectory_file = os.path.join(
                self.sample_directory,
                f'{self.sample_counter:d}_{optimizer_tag:s}.traj')

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.sample_dataset = data.DataSet(
            self.sample_data_file,
            data_file_format=self.sample_data_file_format,
            data_load_properties=self.sample_properties,
            data_unit_properties=self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

    def __str__(self):
        """
        Return class descriptor
        """
        return "Sampler class"

    def read_next_system(self):
        """
        Iterator to read next sample system and return as ASE atoms object
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
        for system, system_format in zip(
                self.sample_systems, self.sample_systems_format):

            # Check for ASE Atoms object or read system file
            if utils.is_ase_atoms(system):
                yield (system, 1, system)
            else:
                isys=0
                while True:
                    try:
                        system_i = ase.io.read(
                            system, index=isys, format=system_format)
                        yield (system_i, isys + 1, system)
                    except (StopIteration, AssertionError):
                        break
                    else:
                        isys += 1

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
                isys=0
                while True:
                    try:
                        sample_systems_atoms.append(
                            ase.io.read(
                                system, index=isys, format=system_format))
                    except (StopIteration, AssertionError):
                        break
                    else:
                        isys += 1
        
        return sample_systems_atoms

    def assign_calculator(
        self,
        sample_system: ase.Atoms,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Assign calculator to a list of sample ASE Atoms objects

        Parameters
        ----------
        sample_system : ase.Atoms
            ASE Atoms object to assign the calculator
        sample_calculator : (str, object), optional, default None
            ASE calculator object or string of an ASE calculator class
            name to assign to the sample systems
        sample_calculator_args : dict, optional, default None
            Dictionary of keyword arguments to initialize the ASE
            calculator
        """

        # Check calculator input
        if sample_calculator is None:
            sample_calculator = self.sample_calculator
        if sample_calculator_args is None:
            sample_calculator_args = self.sample_calculator_args

        if sample_calculator is None:
            
            sample_calculator = self.sample_calculator
            sample_calculator_tag = self.sample_calculator_tag
        
        else:

            # Get ASE calculator
            sample_calculator, sample_calculator_tag = (
                interface.get_ase_calculator(
                    sample_calculator,
                    sample_calculator_args)
                )

            # Check requested system properties
            self.check_properties(sample_calculator)

        # Assign ASE calculator
        sample_system.set_calculator(sample_calculator)

        return sample_system

    def check_properties(
        self, 
        sample_calculator: Optional[object] = None):
        """
        Check requested sample properties and units with implemented properties
        of the calculator
        """
        
        # Check calculator input
        if sample_calculator is None:
            sample_calculator = self.sample_calculator
        
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
        self.sample_unit_properties = {
            prop: interface.ase_calculator_units.get(prop)
            for prop in self.sample_properties}
        if 'positions' not in self.sample_unit_properties:
            self.sample_unit_properties['positions'] = 'Ang'
        self.sample_unit_positions = self.sample_unit_properties['positions']

        return

    def get_info(self):
        """
        Dummy function for sampling parameter dictionary
        """
        return {            
            'sample_data_file': self.sample_data_file,
            'sample_directory': self.sample_directory,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'sample_data_overwrite': self.sample_data_overwrite,
        }

    def run(
        self,
        **kwargs
    ):
        """
        Perform sampling of all sample systems or a selection of them.
        """

        ################################
        # # # Check Sampling Input # # #
        ################################

        # Collect sampling parameters
        config_sample_tag = f'{self.sample_counter}_{self.sample_tag}'
        config_sample = {
            config_sample_tag: self.get_info()
            }

        # Update configuration file with sampling parameters
        if 'sampler_schedule' in self.config:
            config_sample = {
                **self.config['sampler_schedule'],
                **config_sample,
                }
        self.config.update({
            'sampler_schedule': config_sample,
            'sample_counter': self.sample_counter
            })

        # Increment sample counter
        self.sample_counter += 1

        # Print sampling overview
        msg = f"Perform sampling method '{self.sample_tag:s}' on systems:\n"
        for isys, system in enumerate(self.sample_systems):
            msg += f" {isys + 1:3d}. '{system:s}'\n"
        logger.info(f"INFO:\n{msg:s}")

        ###############################
        # # # Perform MD Sampling # # #
        ###############################

        # Iterate over systems
        for (system, isys, source) in self.read_next_system():
            
            # Assign calculator
            system = self.assign_calculator(system)

            # If requested, perform structure optimization
            if self.sample_systems_optimize:

                # Perform structure optimization
                self.ase_optimizer(
                    system,
                    logfile=self.ase_optimizer_log_file,
                    trajectory=self.ase_optimizer_trajectory_file,
                    ).run(
                        fmax=self.sample_systems_optimize_fmax)

            # Start sampling
            Nsamples = self.run_system(system, **kwargs)
            
            # Print sampling info
            msg = f"Sampling method '{self.sample_tag:s}' complete for system "
            msg += f"of index {isys:d} from '{source}!'\n"
            msg += f"{Nsamples:d} samples written to "
            msg += f"'{self.sample_data_file:s}'.\n"
            logger.info(f"INFO:\n{msg:s}")

    def run_system(self, system):
        """
        Apply sample calculator on system input and write properties to 
        database.
        
        Parameters
        ----------
        system: ase.Atoms
            ASE Atoms object serving as initial frame
        """
        # Initialize stored sample counter
        self.Nsample = 0
        
        # Compute system properties
        self.sample_calculator.calculate(
            system,
            properties=self.sample_properties)
        
        # Store results
        self.save_properties(system)

        return self.Nsample

    def get_properties(self, system):
        """
        Collect system properties and calculator results
        """

        return interface.get_ase_properties(system, self.sample_properties)

    def save_properties(self, system):
        """
        Save system properties
        """
        
        system_properties = self.get_properties(system)
        self.sample_dataset.add_atoms(system, system_properties)
        self.Nsample += 1
