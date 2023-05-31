import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import optimize
from ase import vibrations
from ase.visualize import view

from .. import data
from .. import model
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
    Conformation Sampler class for generation of reference structures.
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        sample_directory: Optional[str] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        sample_systems_optimize: Optional[bool] = None,
        sample_systems_optimize_fmax: Optional[float] = None,
        sample_ref_calculator: Optional[Union[str, object]] = None,
        sample_ref_calculator_args: Optional[Dict[str, Any]] = None,
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
            Working directory where to store ASE calculator files,
            ASE trajectory files and/or model calulator files.
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
            class object or a string with avaiable ASE calculator classes.
        sample_calculator_args: dict, optional, default {}
            In case of string type input for 'sample_calculator', this
            dictionary is passed as keyword arguments at the initialization
            of the ASE calculator.
        sample_systems_optimize: bool, optional, default False
            Instruction flag, if the initial system coordinate shall be
            optimized using the ASE calculator defined by 'sample_calculator'.
        sample_ref_calculator: (str, callable object), optional, default None
            Same as 'sample_calculator' but only used for the final reference
            data computation of the final selection of reference structures
        sample_ref_calculator_args: dict, optional, default {}
            Same as 'sample_calculator_args' but for the final reference
            data ASE calculator class if defined by a string input.

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
                f"WARNING:\nNo input in 'sample_systems' is given!\n" +
                f"Please provide either a chemical structure file or " +
                f"an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

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
                f"{len(self.sample_systems_format):d}, respectivelly.")

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

        # If requested, perform structure optimization
        if self.sample_systems_optimize:

            # Assign ASE optimizer
            ase_optimizer = optimize.BFGS

            # Iterate over systems
            for system in self.sample_systems_atoms:

                # Perform structure optimization
                ase_optimizer(system).run(
                    fmax=self.sample_systems_optimize_fmax)


    def normal_mode_sampling(
        self,
        sample_file: Optional[str] = None,
        **kwargs
    ):
        """
        Test function for normal mode sampling
        """

        # 1. Normal mode analysis
        self.nms_systems_frequencies = []
        self.nms_systems_energies = []
        self.nms_systems_modes = []
        for isys, system in enumerate(self.sample_systems_atoms):

            ase_vibrations = vibrations.Vibrations(
                system,
                name=os.path.join(self.sample_directory, f"vib{isys:d}"))
            ase_vibrations.run()
            ase_vibrations.summary()
            [ase_vibrations.write_mode(n) for n in range(12)]

            self.nms_systems_frequencies.append(
                ase_vibrations.get_frequencies())
            self.nms_systems_energies.append(
                ase_vibrations.get_energies())
            self.nms_systems_modes.append([
                ase_vibrations.get_mode(imode)
                for imode, _ in enumerate(self.nms_systems_energies[-1])])

            # TODO ASE: Imaginary frequencies with imag(freq) > 1.e-8
            # TODO ASE: Non-vibrational modes with abs(energy) > 1.e-5

        # 2. Detect translational normal modes

        pass


