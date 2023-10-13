import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import itertools

import ase
from ase import optimize
from ase import units

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from ase.io.trajectory import Trajectory

from .. import data
from .. import model
from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MDSampler']


class MDSampler(sample.Sampler):
    """
    Molecular Dynamics Sampler class
    """
    
    def __init__(
        self,
        md_temperature: Optional[float] = None,
        md_time_step: Optional[float] = None,
        md_simulation_time: Optional[float] = None,
        md_save_interval: Optional[float] = None,
        md_langevin_friction: Optional[float] = None,
        md_equilibration_time: Optional[float] = None,
        md_initial_velocities: Optional[bool] = None,
        md_initial_temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize MD sampling class
        
        Parameters
        ----------

        md_temperature: float, optional, default 300
            Target temperature in Kelvin of the MD simulation controlled by a
            Langevin thermostat
        md_time_step: float, optional, default 1.0 (1 fs)
            MD Simulation time step in fs
        md_simulation_time: float, optional, default 1E5 (100 ps)
            Total MD Simulation time in fs
        md_save_interval: int, optional, default 10
            MD Simulation step interval to store system properties of 
            the current frame to dataset.
        md_langevin_friction: float, optional, default 0.01
            Langevin thermostat friction coefficient in Kelvin. Generally 
            within the magnitude of 1E-2 (fast heating/cooling) to 1E-4 (slow)
        md_equilibration_time: float, optional, default 0 (no equilibration)
            Total MD Simulation time in fs for a equilibration run prior to
            the production run.
        md_initial_velocities: bool, optional, default False
            Instruction flag if initial atom velocities are assigned with
            respect to a Maxwell-Boltzmann distribution at temperature
            'md_initial_temperature'.
        md_initial_temperature: float, optional, default 300
            Temperature for initial atom velocities according to a Maxwell-
            Boltzmann distribution.
        
        Returns
        -------
        object
            Molecular Dynamics Sampler class object
        """
        
        super().__init__(**kwargs)
        
        ################################
        # # # Check MD Class Input # # #
        ################################
        
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
                item = self.config.get(arg)

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
        #self.config.update(config_update)
        
        # Sampler class label
        self.sample_tag = 'md'
        
        # Check sample data file
        if self.sample_data_file is None:
            self.sample_data_file = os.path.join(
                self.sample_directory, 
                f'{self.sample_counter:d}_{self.sample_tag:s}.db')
        elif not utils.is_string(self.sample_data_file):
            raise ValueError(
                f"Sample data file 'sample_data_file' must be a string " +
                f"of a valid file path but is of type " + 
                f"'{type(self.sample_data_file)}'.")
        
        # Define MD log and trajectory file path
        self.md_log_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.md_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')
        
        # Check sample properties for energy and forces properties which are 
        # required for MD sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')
        
        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################
        
        self.md_dataset = data.DataSet(
            self.sample_data_file,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=True)
        
        return
    
    def get_info(self):
        
        return {
            'sample_directory': self.sample_directory,
            'sample_data_file': self.sample_data_file,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'md_temperature': self.md_temperature,        
            'md_time_step': self.md_time_step,
            'md_simulation_time': self.md_simulation_time,
            'md_save_interval': self.md_save_interval,
            'md_langevin_friction': self.md_langevin_friction,
            'md_equilibration_time': self.md_equilibration_time,
            'md_initial_velocities': self.md_initial_velocities,
            'md_initial_temperature': self.md_initial_temperature,
        }
    
    def run_system(self, system):
        """
        Perform MD Simulation with the sample system.
        """

        # Set initial atom velocities if requested
        if self.md_initial_velocities:
            MaxwellBoltzmannDistribution(
                system, 
                temperature_K=self.md_initial_temperature)
        
        # Initialize MD simulation propagator
        md_dyn = Langevin(
            system, 
            timestep=self.md_time_step*units.fs,
            temperature_K=self.md_temperature,
            friction=self.md_langevin_friction,
            logfile=self.md_log_file,
            loginterval=self.md_save_interval)
        
        # Perform MD equilibration simulation if requested
        if (
                self.md_equilibration_time is not None 
                and self.md_equilibration_time > 0.
            ):
            
            # Run equilibration simulation
            md_equilibration_step = round(
                self.md_equilibration_time/self.md_time_step)
            md_dyn.run(md_equilibration_step)
            
        # Attach system properties saving function
        md_dyn.attach(
            self.save_properties,
            interval=self.md_save_interval,
            system=system)

        # Attach trajectory
        self.md_trajectory = Trajectory(
            self.md_trajectory_file, atoms=system, 
            mode='a', properties=self.sample_properties)
        md_dyn.attach(
            self.write_trajectory, 
            interval=self.md_save_interval,
            system=system)

        # Run MD simulation
        md_simulation_step = round(
            self.md_simulation_time/self.md_time_step)
        md_dyn.run(md_simulation_step)
        

    def save_properties(self, system):
        """
        Save system properties
        """
        
        system_properties = self.get_properties(system)
        self.md_dataset.add_atoms(system, system_properties)
        

    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """
        
        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.md_trajectory.write(system_noconstraint)
