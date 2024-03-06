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
from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MDSampler']


class MDSampler(sample.Sampler):
    """
    Molecular Dynamics Sampler class

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

    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'md_temperature':               300.,
        'md_time_step':                 1.,
        'md_simulation_time':           1.E5,
        'md_save_interval':             100,
        'md_langevin_friction':         1.E-2,
        'md_equilibration_time':        None,
        'md_initial_velocities':        False,
        'md_initial_temperature':       300.,
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'md_temperature':               [utils.is_numeric],
        'md_time_step':                 [utils.is_numeric],
        'md_simulation_time':           [utils.is_numeric],
        'md_save_interval':             [utils.is_integer],
        'md_langevin_friction':         [utils.is_numeric],
        'md_equilibration_time':        [utils.is_numeric],
        'md_initial_velocities':        [utils.is_bool],
        'md_initial_temperature':       [utils.is_numeric],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
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

        """
        
        # Sampler class label
        self.sample_tag = 'md'


        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )

        ################################
        # # # Check MD Class Input # # #
        ################################
        
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

        # Check sample properties for energy and forces properties which are 
        # required for MD sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        return
    
    def get_info(self):
        """
        Get sampler information

        Returns
        -------
        dict
            Sampler information dictionary
        """
        
        info = super().get_info()
        info.update({
            'md_temperature': self.md_temperature,        
            'md_time_step': self.md_time_step,
            'md_simulation_time': self.md_simulation_time,
            'md_save_interval': self.md_save_interval,
            'md_langevin_friction': self.md_langevin_friction,
            'md_equilibration_time': self.md_equilibration_time,
            'md_initial_velocities': self.md_initial_velocities,
            'md_initial_temperature': self.md_initial_temperature,
            })
        
        return info

    def run_system(self, system):
        """
        Perform MD Simulation with the sample system.

        Parameters
        ----------
        system: ase.Atoms
            ASE Atoms object serving as initial frame
        """

        # Initialize stored sample counter
        self.Nsample = 0

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
            logfile=self.sample_log_file,
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
        if self.sample_save_trajectory:
            self.md_trajectory = Trajectory(
                self.sample_trajectory_file, atoms=system, 
                mode='a', properties=self.sample_properties)
            md_dyn.attach(
                self.write_trajectory, 
                interval=self.md_save_interval,
                system=system)

        # Run MD simulation
        md_simulation_step = round(
            self.md_simulation_time/self.md_time_step)
        md_dyn.run(md_simulation_step)
        
        return self.Nsample

    def save_properties(self, system):
        """
        Save system properties
        """
        
        system_properties = self.get_properties(system)
        self.sample_dataset.add_atoms(system, system_properties)
        self.Nsample += 1
        

    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """
        
        system_noconstraint = system.copy()
        system_noconstraint.calc = system.calc
        system_noconstraint.set_constraint()
        self.md_trajectory.write(system_noconstraint)
