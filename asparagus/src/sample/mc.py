import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import units
from ase.md.md import MolecularDynamics

from ase.io.trajectory import Trajectory

from .. import data
from .. import model
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

    """

    def __init__(
            self,
            mc_data_file: Optional[str] = None,
            mc_temperature: Optional[float] = None,
            mc_time_step: Optional[float] = None,
            mc_simulation_time: Optional[float] = None,
            mc_equilibration_time: Optional[float] = None,
            mc_max_displacement: Optional[float] = None,
            **kwargs,
    ):

        super().__init__(**kwargs)

        ################################
        # # # Check MC Class Input # # #
        ################################

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        # config_update = {}
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
            # config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        # self.config.update(config_update)

        # Sampler class label
        self.sample_tag = 'mc'

        # Check sample data file
        if self.mc_data_file is None:
            self.mc_data_file = os.path.join(
                self.sample_directory,
                f'{self.sample_counter:d}_{self.sample_tag:s}.db')
        elif not utils.is_string(self.mc_data_file):
            raise ValueError(
                f"Sample data file 'mc_data_file' must be a string " +
                f"of a valid file path but is of type " +
                f"'{type(self.mc_data_file)}'.")

        # Define MC log file path
        self.mc_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.mc_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')
        
        # Check sample properties for energy and forces properties which are
        # required for MC sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.mc_dataset = data.DataSet(
            self.mc_data_file,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=True)

        return

    def get_info(self):

        return {
            'sample_directory': self.sample_directory,
            # 'sample_data_file': self.sample_data_file,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'mc_data_file': self.mc_data_file,
            'mc_temperature': self.mc_temperature,
            'mc_time_step': self.mc_time_step,
            'mc_simulation_time': self.mc_simulation_time,
            'mc_save_interval': self.mc_save_interval,
            'mc_equilibration_time': self.mc_equilibration_time,
        }

    def run_system(self, system, save_trajectory=False):
        """
        Perform a very simple MC Simulation using the Metropolis algorithm with the sample system.
        """

        # Set up system
        initial_system = system.copy()
        initial_system.set_calculator(self.sample_calculator)
        
        # Temperature parameter
        self.beta = 1.0 / (units.kB * self.mc_temperature)
        
        # Perform MC equilibration simulation if requested
        if (
                self.mc_equilibration_time is not None
                and self.mc_equilibration_time > 0.
        ):
            
            eq_steps = int(self.mc_equilibration_time / self.mc_time_step)
            traj_eq = self.monte_carlo_steps(initial_system, eq_steps)
            
            #  Set the equilibrated system as the new initial system
            initial_system = traj_eq[-1]

        # Initialize trajectory
        self.mc_trajectory = Trajectory(
            self.md_trajectory_file, atoms=system, 
            mode='a', properties=self.sample_properties)
        
        # Perform MC simulation
        traj_steps = int(self.mc_simulation_time / self.mc_time_step)
        traj = self.monte_carlo_steps(initial_system, traj_steps)

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
        system
        steps

        Returns
        -------

        """

        current_energy = system.get_potential_energy()
        
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
            if np.random.rand() < threshold:

                # Set the new position of the atom
                current_energy = new_energy
                self.save_properties(system)
                self.write_trajectory(system)

            else:
                
                # Reset the position of the atom
                system[atom].position = old_position

        return trajectory


    def save_properties(self, system):
        """
        Save system properties
        """

        system_properties = self.get_properties(system)
        self.mc_dataset.add_atoms(system, system_properties)

    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """
        
        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.mc_trajectory.write(system_noconstraint)

# class MonteCarlo(MolecularDynamics):
#
#     '''
#     This class is inspired from the ase langevin dynamics class
#      TODO: Finish this!
#
#     '''
#
#     def __init__(self,atoms,timestep,temperature,max_displacement=None,
#                  trajectory=None,logfile=None,
#                  loginterval=1,append_trajectory=False,**kwargs):
#
#         self.temperature = temperature
#         self.beta = 1.0 / (units.kB * self.temperature)
#         self.max_displacement = max_displacement
#
#         MolecularDynamics.__init__(self, atoms, timestep, trajectory,logfile, loginterval,
#                                        append_trajectory=append_trajectory)
#         self.updatevars()
#
#     def todict(self):
#         d = MolecularDynamics.todict(self)
#         d.update({'temperature_K': self.temp / units.kB})
#         return d
#
#     def set_temperature(self, temperature=None, temperature_K=None):
#         self.temp = units.kB * self._process_temperature(temperature,
#                                                          temperature_K, 'eV')
#         self.updatevars()
#
#     def set_timestep(self,timestep):
#         self.dt = timestep
#         self.updatevars()
#
#     def updatevars(self):
#         dt = self.dt
#         T = self.temp
#
#     def step(self):
#
#         atoms = self.atoms
#
#         # Randomly choose an atom
#         i = np.random.randint(len(atoms))
#         orig_pos = atoms[i].positions.copy()
#
#         # Randomly displace the atom
#         displacement = np.random.uniform(-self.max_displacement,self.max_displacement,3)
#         atoms[i].positions += displacement
