import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import itertools

import ase
from ase import units

from ase import vibrations
from ase.constraints import FixAtoms

from ase.io.trajectory import Trajectory

from .. import data
from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['NormalModeScanner', 'NormalModeSampler']


class NormalModeScanner(sample.Sampler):
    """
    Normal Mode Scanning class
    """

    def __init__(
        self,
        nms_harmonic_energy_step: Optional[float] = None,
        nms_energy_limits: Optional[Union[float, List[float]]] = None,
        nms_number_of_coupling: Optional[int] = None,
        nms_limit_of_steps: Optional[int] = None,
        nms_limit_com_shift: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize Normal Mode Scanning class

        Parameters
        ----------

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
            If a list with two numeric values are given, the first two are the
            lower and upper potential energy limit, respectively.
        nms_number_of_coupling: int, optional, default 2
            Maximum number of coupled normal mode displacements to sample
            the system conformations.
        nms_limit_of_steps: int, optional, default 10
            Maximum limit of coupled normal mode displacements in one direction
            to sample the system conformations.
        nms_limit_com_shift: float, optional, default 0.1 Angstrom
            Center of mass shift threshold to identify translational normal
            modes from vibrational (and rotational). Normalized Normal modes
            with a center of mass shift larger than the threshold are not
            considered in the normal mode scan.

        Returns
        -------
        object
            Normal Mode Scanning class object
        """

        # Sampler class label
        self.sample_tag = 'nmscan'

        # Initialize parent class
        super().__init__(sample_tag=self.sample_tag, **kwargs)

        #################################
        # # # Check NMS Class Input # # #
        #################################

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
                item = self.config.get(arg)

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
        self.config.update(config_update)

        # Define MD log and trajectory file path
        self.nms_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.nms_trajectory_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')

        # Check sample properties for energy property which is required for
        # normal mode scanning
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')

        # Check potential energy limits
        if utils.is_numeric(self.nms_energy_limits):
            self.nms_energy_limits = [
                -abs(self.nms_energy_limits), abs(self.nms_energy_limits)]

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.nms_dataset = data.DataSet(
            self.sample_data_file,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

        return

    def get_info(self):

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
            'nms_harmonic_energy_step': self.nms_harmonic_energy_step,
            'nms_energy_limits': self.nms_energy_limits,
            'nms_number_of_coupling': self.nms_number_of_coupling,
            'nms_limit_com_shift': self.nms_limit_com_shift,
            'nms_limit_of_steps': self.nms_limit_of_steps,
        }

    def run_system(
        self,
        system: object,
        nms_indices: Optional[List[int]] = None,
        nms_clean: Optional[bool] = False,
        **kwargs
    ):
        """
        Perform Normal Mode Scanning on the sample system.

        Parameters
        ----------

        system: ase.Atoms object
            ASE atoms object to perform Normal Mode Scanning on.
        nms_indices: list[int], optional, default None
            List of atom indices to include in normal mode analysis.
            If none, indices if a full list of atom indices with length ot the
            atom number of the system.
            Atom indices from atoms constraint by FicAtoms are reomved from
            index list and the normal mode analysis.
        nms_clean: bool, optional, default False
            If True, checkpoint files for atom displacement calculations
            in 'sample_directory'/vib will be deleted.
            Else, results from avaible  checkpoint files will be used.
        """

        # Compute initial state properties
        # TODO Special calculate property function to handle special properties
        # not supported by ASE such as, e.g., charge, hessian, etc.
        self.sample_calculator.calculate(
            system,
            properties=self.sample_properties)
        system_init_potential = system._calc.results['energy']

        # Add initial state properties to dataset
        system_properties = self.get_properties(system)
        self.nms_dataset.add_atoms(system, system_properties)

        # Attach to trajectory
        self.nms_trajectory = Trajectory(
            self.nms_trajectory_file, atoms=system,
            mode='a', properties=self.sample_properties)
        self.write_trajectory(system)

        # Get non-fixed atoms indices
        if nms_indices is None:
            indices = np.arange(system.get_global_number_of_atoms(), dtype=int)
        else:
            indices = np.array(nms_indices, dtype=int)
        for constraint in system.constraints:
            if isinstance(constraint, FixAtoms):
                indices = [
                    idx for idx in indices
                    if idx not in constraint.index]
        indices = np.array(indices)

        # Prepare system parameter
        Natoms = len(indices)
        Nmodes = 3*Natoms

        # Perform numerical normal mode analysis
        ase_vibrations = vibrations.Vibrations(
            system,
            indices=indices,
            name=os.path.join(self.sample_directory, "vib"))
        if nms_clean:
            ase_vibrations.clean()
        ase_vibrations.run()
        ase_vibrations.summary()

        # (Trans. + Rot. + ) Vibrational frequencies in cm**-1
        system_frequencies = ase_vibrations.get_frequencies()

        # (Trans. + Rot. + ) Vibrational modes normalized to 1
        system_modes = np.array([
            ase_vibrations.get_mode(imode)[indices].reshape(Natoms, 3)
            / np.sqrt(np.sum(
                ase_vibrations.get_mode(imode)[indices].reshape(Natoms, 3)**2))
            for imode in range(Nmodes)])

        # Reduced mass per mode (in amu)
        system_redmass = np.array([
            1./np.sum(
                system_modes[imode]**2/system.get_masses()[indices].reshape(
                    Natoms, 1))
            for imode in range(Nmodes)])

        # Force constant per mode (in eV/Angstrom**2)
        system_forceconst = (
            4.0*np.pi**2*(np.abs(system_frequencies)*1.e2*units._c)**2
            * system_redmass*units._amu*units.J*1.e-20)

        # Compute and store equilibrium positions and center of mass,
        # moments of inertia and principle axis of inertia
        system_init_positions = system.get_positions()
        system_init_com = system[indices].get_center_of_mass()

        # Compute and compare same quantities for displaced system
        system_com_shift = np.zeros(Nmodes, dtype=float)

        for imode, mode in enumerate(system_modes):

            # COM shift
            system_mode_positions = system_init_positions.copy()
            system_mode_positions[indices] += mode
            system.set_positions(system_mode_positions, apply_constraint=False)
            system_displ_com = system[indices].get_center_of_mass()
            system_com_shift[imode] = np.sqrt(np.sum(
                (system_init_com - system_displ_com)**2))

        # Vibrational modes are assumed with center of mass shifts smaller
        # than 1.e-1 Angstrom displacement
        system_vib_modes = system_com_shift < self.nms_limit_com_shift

        # Displacement factor for energy step (in eV)
        system_displfact = np.sqrt(
            3*self.nms_harmonic_energy_step/system_forceconst)

        # Add normal mode analysis results to log file
        msg = "\nStart Normal Mode Scanning at system: "
        if self.sample_data_file is None:
            msg += f"{system.get_chemical_formula():s}\n"
        else:
            msg += f"{self.sample_data_file:s}\n"
        msg += " Index | Frequency (cm**-1) | Vibration (CoM displacemnt)\n"
        msg += "---------------------------------------------------------\n"
        for ivib, freq in enumerate(system_frequencies):
            msg += f" {ivib + 1:5d} | {freq:18.2f} |"
            if system_vib_modes[ivib]:
                msg += f"   x   ({system_com_shift[ivib]:2.1e})\n"
            else:
                msg += f"       ({system_com_shift[ivib]:2.1e})\n"
        with open(self.nms_log_file, 'a') as flog:
            flog.write(msg)

        # Iterate over number of normal mode combinations
        steps = np.arange(1, self.nms_limit_of_steps + 1, 1)
        vib_modes = np.where(system_vib_modes)[0]
        for icomp in range(1, self.nms_number_of_coupling + 1):

            # Prepare sign combinations
            all_signs = np.array(list(
                itertools.product((-1, 1), repeat=icomp)))

            all_steps = np.array(list(
                itertools.product(steps, repeat=icomp)))
            Nsteps = all_steps.shape[0]

            # Iterate over vib. normal mode indices and their combinations
            for imodes in itertools.combinations(vib_modes, icomp):

                # Iterate over sign combinations
                for isign, signs in enumerate(all_signs):

                    # Iterate through steps
                    istep = 0
                    done = False
                    while not done:

                        # Get current step size
                        step_size = np.array(all_steps[istep])

                        # Get normal mode elongation step
                        current_step = np.array(signs)*step_size

                        # Set elongation step on initial system positions
                        current_step_positions = system_init_positions.copy()
                        for imode, modei in enumerate(imodes):
                            current_step_positions[indices] += (
                                system_displfact[modei]*current_step[imode]
                                * system_modes[modei])
                        system.set_positions(
                            current_step_positions,
                            apply_constraint=False)

                        # Compute observables and check potential threshold
                        try:

                            self.sample_calculator.calculate(
                                system,
                                properties=self.sample_properties)
                            current_properties = system._calc.results
                            current_potential = current_properties['energy']

                            converged = True

                            if current_potential < system_init_potential:
                                threshold_reached = (
                                    (current_potential - system_init_potential)
                                    < self.nms_energy_limits[0])
                            else:
                                threshold_reached = (
                                    (current_potential - system_init_potential)
                                    > self.nms_energy_limits[1])

                        except ase.calculators.calculator.CalculationFailed:

                            converged = False
                            threshold_reached = True

                        # Add to dataset
                        if converged:
                            system_properties = self.get_properties(system)
                            self.nms_dataset.add_atoms(
                                system, system_properties)

                        # Attach to trajectory
                        self.write_trajectory(system)

                        # Check energy threshold
                        if threshold_reached:

                            # Check for next suitable step size index and avoid
                            # combination of step sizes which were already
                            # above the energy threshold before.
                            istep += 1
                            for jstep in range(istep, Nsteps):
                                if np.any(
                                    np.array(all_steps[jstep]) < step_size
                                ):
                                    istep = jstep
                                    break
                            else:
                                done = True

                        else:

                            # Increment step size index
                            istep += 1

                        # Check step size progress
                        if done or istep >= Nsteps:

                            # Update log file
                            msg = "Vib. modes: ("
                            for imode, isign in zip(imodes, signs):
                                if isign > 0:
                                    msg += f"+{imode + 1:d}, "
                                else:
                                    msg += f"-{imode + 1:d}, "
                            msg += f") - {istep:4d} steps added\n"
                            with open(self.nms_log_file, 'a') as flog:
                                flog.write(msg)

                            # Set flag in case maximum Nsteps is reached
                            done = True

        return

    def write_trajectory(
        self,
        system
    ):
        """
        Write current image to trajectory file but without constraints
        """

        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.nms_trajectory.write(system_noconstraint)


class NormalModeSampler(sample.Sampler):
    """
        Normal Mode Sampling class.

        This is the vanilla version of the normal mode class.

    """

    def __init__(
        self,
        nms_temperature: Optional[float] = None,
        nms_nsamples: Optional[int] = None,
        **kwargs
    ):

        # Sampler class label
        self.sample_tag = 'nmsamp'

        # Initialize parent class
        super().__init__(sample_tag=self.sample_tag, **kwargs)

        #################################
        # # # Check NMS Class Input # # #
        #################################

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                'self', 'config', 'config_update', 'kwargs', '__class__'
            ]:
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
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)

        # Update global configuration dictionary
        self.config.update(config_update)

        # Define MD log and trajectory file path
        self.nms_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.log')
        self.nms_trajectory_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')

        # Check sample properties for energy property which is required for
        # normal mode scanning
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.nms_dataset = data.DataSet(
            self.sample_data_file,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

        return

    def get_info(self):

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
            'nms_temperature': self.nms_temperature,
            'nms_nsamples': self.nms_nsamples,
        }

    def R(self, fct, nmodes, T=300):

        random_num = np.random.uniform(size=nmodes)**2
        sign = [-1 if i < 0.5 else 1 for i in random_num]
        fix_fcts = [0.05 if i < 0.05 else i for i in fct]
        R = []

        for i, j in enumerate(fix_fcts):
            R_i = sign[i] * np.sqrt((3*random_num[i]*units.kB*T)/j)
            R.append(R_i)

        return np.array(R)

    def new_coord(self, nmodes, vib_disp, mass_sqrt, fcts, T=300):

        Rx = self.R(nmodes, fcts, T)
        new_disp = []

        for i, j in enumerate(vib_disp):
            disp_i = mass_sqrt[i] * Rx[i] * j
            new_disp.append(disp_i)

        return np.sum(new_disp, axis=0)

    def save_properties(self, system):
        """
        Save system properties
        """

        system_properties = self.get_properties(system)
        self.nms_dataset.add_atoms(system, system_properties)

    def run(
        self,
        system: object,
        nms_indices: Optional[List[int]] = None,
    ):
        """
        Perform Normal Mode Sampling on the sample system.
        """

        # Compute initial state properties
        self.sample_calculator.calculate(
            system,
            properties=self.sample_properties)

        # Add initial state properties to dataset
        system_properties = self.get_properties(system)
        self.nms_dataset.add_atoms(system, system_properties)

        # Attach to trajectory
        self.nms_trajectory = Trajectory(
            self.nms_trajectory_file, atoms=system,
            mode='a', properties=self.sample_properties)
        self.write_trajectory(system)

        # Get non-fixed atoms indices
        if nms_indices is None:
            indices = range(system.get_global_number_of_atoms())
        else:
            indices = np.array(nms_indices, dtype=int)
        for constraint in system.constraints:
            if isinstance(constraint, FixAtoms):
                indices = [
                    idx for idx in indices
                    if idx not in constraint.index]
        indices = np.array(indices)

        # Prepare system parameter
        Natoms = len(indices)
        Nmodes = 3*Natoms

        # Perform numerical normal mode analysis
        ase_vibrations = vibrations.Vibrations(
            system,
            indices=indices,
            name=os.path.join(self.sample_directory, "vib")
            )
        ase_vibrations.clean()
        ase_vibrations.run()
        ase_vibrations.summary()

        # (Trans. + Rot. + ) Vibrational frequencies in cm**-1
        system_frequencies = ase_vibrations.get_frequencies()

        # (Trans. + Rot. + ) Vibrational modes normalized to 1
        system_modes = np.array([
            ase_vibrations.get_mode(imode)[indices].reshape(Natoms, 3)
            / np.sqrt(
                np.sum(
                    ase_vibrations.get_mode(imode)[indices].reshape(
                        Natoms, 3)**2
                    )
                )
            for imode in range(Nmodes)])

        # Reduced mass per mode (in amu)
        system_redmass = np.array([
            1. / np.sum(
                system_modes[imode]**2/system.get_masses()[indices].reshape(
                    Natoms, 1))
            for imode in range(Nmodes)])

        # Force constant per mode (in eV/Angstrom**2)
        system_forceconst = (
            4.0*np.pi**2*(np.abs(system_frequencies)*1.e2*units._c)**2
            * system_redmass*units._amu*units.J*1.e-20)

        # Compute and store equilibrium positions and center of mass,
        # moments of inertia and principle axis of inertia
        system_init_positions = system.get_positions()
        for _ in range(self.nmsamp_nsamples):

            new_position = system_init_positions.copy()
            new_position[indices] += self.new_coord(
                Nmodes, system_modes, system_redmass, system_forceconst,
                self.nmsamp_temperature)
            system.set_positions(new_position)

            try:

                self.sample_calculator.calculate(
                    system,
                    properties=self.sample_properties)
                self.save_properties(system)

            # TODO Add correct exception error
            except:

                print('This configuration is not stable')
                pass

    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """

        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.md_trajectory.write(system_noconstraint)
