import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import units
from ase.md.md import MolecularDynamics

from ase.io.trajectory import Trajectory

from .. import data
from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MCSampler']


class MCSampler(sample.Sampler):
    """
    A very simple Monte Carlo (MC) sampler class.
    Uses the Metropolis algorithm to generate samples for a molecule.

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    mc_temperature: float, optional, default 300
        Temperature of the MC simulation in Kelvin.
    mc_steps: integer, optional, default 1000
        Number of MC simulations steps
    mc_save_interval: int, optional, default 1
        Step interval to store system properties of the current frame.
    mc_max_displacement: float
        Maximum displacement of the MC simulation in Angstrom.
    
    Returns
    -------
    object
        Monta-Carlo Sampler class object
    """

    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'mc_temperature':               300.,
        'mc_steps':                     1000,
        'mc_max_displacement':          0.1,
        'mc_save_interval':             1,
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'mc_temperature':               [utils.is_numeric],
        'mc_steps':                     [utils.is_numeric],
        'mc_max_displacement':          [utils.is_numeric],
        'mc_save_interval':             [utils.is_integer],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        mc_temperature: Optional[float] = None,
        mc_steps: Optional[int] = None,
        mc_max_displacement: Optional[float] = None,
        mc_save_interval: Optional[int] = None,
        **kwargs,
    ):

        # Sampler class label
        self.sample_tag = 'mc'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )
        
        ################################
        # # # Check MC Class Input # # #
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

        # Define MC log file path
        self.mc_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.mc_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')
        
        # Check sample properties for energy properties which are required for 
        # MC sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')

        return

    def get_info(self):

        """
        Returns a dictionary with the information of the MC sampler.
        
        Returns
        -------
        dict
            Dictionary with the information of the MC sampler.
        """
        
        info = super().get_info()
        info.update({
            'mc_temperature': self.mc_temperature,
            'mc_steps': self.mc_steps,
            'mc_max_displacement': self.mc_max_displacement,
            'mc_save_interval': self.mc_save_interval,
            })

        return info

    def run_system(
        self, 
        system: ase.Atoms, 
        save_trajectory: Optional[bool] = False):
        """
        Perform a very simple MC Simulation using the Metropolis algorithm with
        the sample system.

        Parameters
        ----------
        system: ase.Atoms
            System to be sampled.
        save_trajectory: bool, optional default 'False'
            Save trajectory of the MC simulation.
        """

        # Initialize stored sample counter
        self.Nsample = 0
        
        # Set up system
        initial_system = system.copy()
        initial_system.set_calculator(self.sample_calculator)
        
        # Temperature parameter
        self.beta = 1.0 / (units.kB * self.mc_temperature)

        # Initialize trajectory
        self.mc_trajectory = Trajectory(
            self.mc_trajectory_file, atoms=system,
            mode='a', properties=self.sample_properties)
        
        # Perform MC simulation
        self.monte_carlo_steps(initial_system, self.mc_steps)
        
        return self.Nsample

    def monte_carlo_steps(
        self,
        system,
        steps
    ):
        """
        This does a simple Monte Carlo simulation using the Metropolis
        algorithm.

        In the future we could add more sophisticated sampling methods
        (e.g. MALA or HMC)

        Parameters
        ----------
        system: ase.Atoms
            System to be sampled.
        steps: int
            Number of MC steps to perform.
        """
    
        # Compute initial energy
        current_energy = system.get_potential_energy()
        
        # MC acceptance counter
        Naccept = 0
        
        # Iterate over MC steps
        for step in range(steps):
            
            # First, randomly pick an atom
            atom = np.random.randint(0, len(system))

            # Store its current position
            old_position = system.positions[atom].copy()

            # Propose a new random position for this atom
            displacement = np.random.uniform(
                -self.mc_max_displacement, self.mc_max_displacement, 3)
            system[atom].position += displacement

            # Get the potential energy of the new system
            new_energy = system.get_potential_energy()

            # Metropolis acceptance criterion
            threshold = np.exp(-self.beta * (new_energy - current_energy))
            
            # If accepted 
            if np.random.rand() < threshold:

                # Set the new position of the atom
                current_energy = new_energy
                Naccept += 1
                
                # Store properties
                if not Naccept%self.mc_save_interval:
                    self.save_properties(system)
                    self.write_trajectory(system)
            
            # If not accepted
            else:
                
                # Reset the position of the atom
                system[atom].position = old_position

        return


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
        self.mc_trajectory.write(system_noconstraint)
