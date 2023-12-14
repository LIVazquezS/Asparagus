import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import matplotlib.pyplot as plt
import warnings

import numpy as np

import ase
from ase.io import read, write
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin


from ase.io.trajectory import Trajectory

from .. import interface
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MDP']

class MDP:

    '''

    This class calculates the Minimum Dynamic Path (MDP) [J. Chem. Phys. 150, 074107 (2019)] for a given molecule starting from the Transition state.

    parameters:
    -----------
    atoms: ase.Atoms
        Transition state geometry
    atoms_charge: opt(list)
        Charge of each atom in the system
    config: opt(dict)
        Configuration dictionary for asparagus
    config_file: opt(str)
          Configuration file for asparagus
    model_checkpoint: opt(int)
        Checkpoint file to load the model
    implemented_properties: opt(list)
        Properties to be calculated by the model
    use_neighbor_list: opt(bool)
    Use neighbor list to calculate the forces
    forward: opt(bool)
        Direction of the MDP if True forward, if False backward
    label: opt(str)
        Label for the ASE calculator
    time_step: opt(float)
        Time step for the MD simulation
    langevin: opt(bool)
        If True use Langevin dynamics, if False use Velocity Verlet
    friction: opt(float)
        Friction coefficient for Langevin dynamics only if langevin=True
    temperature: opt(float)
        Temperature for Langevin dynamics only if langevin=True
    eps: opt(float)
        Initial displacement for the MDP
    number_of_steps: opt(int)
        Number of steps for the MDP
    output: opt(str)
        Output file name for the trajectory
    output_file: opt(str)
        Output file name for the log file of energies


    '''

    def __init__(self,atoms=None,
                 atoms_charge=None,
                 config=None,
                 config_file=None,
                 model_checkpoint=None,
                 implemented_properties=None,
                 use_neighbor_list=False,
                 forward=True,
                 label='mdp',
                 time_step=0.1,
                 langevin=False,
                 friction=0.02,
                 temperature=0,
                 eps=0.0005,
                 number_of_steps=4000,
                 output='mdp.traj',
                 output_file=None,
                 **kwargs):

        self.atoms = atoms
        self.eps = eps
        self.number_of_steps = number_of_steps
        self.output = output
        self.forward = forward
        self.langevin = langevin
        self.time_step = time_step


        if self.langevin:
            self.friction = friction
            self.temperature = temperature
        elif self.langevin is True and (friction is None or temperature is None):
            raise ValueError('The friction and temperature are required for Langevin dynamics')
        else:
            self.friction = None
            self.temperature = None

        if self.output_file is None:
            self.output_file = self.output + '.txt'
        elif self.output_file is False:
            pass
        else:
            self.output_file = output_file

        if self.atoms is type(str):
            self.atoms = read(self.atoms)

        if self.atoms is None:
            raise ValueError('The transition state geometry is required')

        ######################################
        # # # Check ASE Calculator Input # # #
        ######################################

        # Assign model parameter configuration library
        if config is None:
            config_data = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_data = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_data.check()

        # Check for empty config dictionary
        if "model_directory" not in config_data:
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
                config_data,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_data.update(
                self.model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_data, **kwargs)
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
        if implemented_properties is not None:
            self.implemented_properties = implemented_properties
        else:
            self.implemented_properties = ['energy', 'hessian', 'forces']

        if 'hessian' not in self.implemented_properties:
            self.implemented_properties.append('hessian')

        if 'forces' not in self.implemented_properties:
            self.implemented_properties.append('forces')

        self.ase_calculator = interface.ASE_Calculator(
            self.model_calculator,
            atoms=self.atoms,
            atoms_charge=atoms_charge,
            implemented_properties=self.implemented_properties,
            use_neighbor_list=use_neighbor_list,
            label=label,
        )

        # Initialize the trajectory file
        self.mdp_trajectory = Trajectory(self.output, 'w', self.atoms, properties=['energy', 'forces'])


    def run_mdp(self,forward=None):
        """
        This function runs the MDP calculation.

        Parameters
        ----------
        forward: opt(bool) If True forward, if False backward. This defines the direction of the momentum of the system.
        Default is None because it is defined in the __init__ function.

        Returns
        -------

        """
        if forward is not None:
            warnings.WarningMessage('The forward parameter is rewroted to %s' % forward)
            self.forward = forward

        initial_system = self.atoms.copy()
        initial_system.set_calculator(self.ase_calculator)

        n_atom = len(initial_system)

        # Calculate the hessian of the transition state
        initial_system.get_potential_energy()
        hessian = np.reshape(initial_system.calc.results['hessian'], (n_atom * 3, n_atom * 3))

        # Calculate the eigenvalues and eigenvectors of the hessian

        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        eigenvectors = eigenvectors.T
        normdisp = eigenvectors[0].reshape(n_atom, 3)

        if self.forward:
            normdisp = normdisp
            initial_system.set_momenta(self.eps * normdisp)
        else:
            normdisp = -normdisp
            initial_system.set_momenta(self.eps * normdisp)

        pos = initial_system.get_positions()

        new_pos = pos + self.eps * normdisp
        initial_system.set_positions(new_pos)

        if self.langevin:
            dyn = Langevin(initial_system, self.time_step * units.fs, self.temperature * units.kB, self.friction)
        else:
            dyn = VelocityVerlet(initial_system, self.time_step * units.fs)

        self.write_trajectory(initial_system)
        self.ase_calculator.implemented_properties.remove('hessian')

        step = []
        eners_pot = []
        eners_kin = []
        eners_tot = []
        for i in range(self.number_of_steps):
            dyn.run(1)
            ener_i = initial_system.get_potential_energy()
            epot = ener_i/n_atom
            ekin = initial_system.get_kinetic_energy()/n_atom
            etot = epot + ekin
            self.write_trajectory(initial_system)
            eners_tot.append(etot)
            eners_pot.append(epot)
            eners_kin.append(ekin)
            step.append(i)

        if self.output_file is not False:
            np.savetxt(self.output_file, np.c_[step,eners_pot,eners_kin,eners_tot],
                       header='Step,Energy Potential,Energy Kinetic,Energy Total')


    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """

        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.mdp_trajectory.write(system_noconstraint)


