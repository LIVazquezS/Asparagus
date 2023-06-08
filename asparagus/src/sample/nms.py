import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import itertools

import ase
from ase import optimize
from ase import units

from ase import vibrations

from .. import data
from .. import model
from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['NormalModeScanner']


class NormalModeScanner(sample.Sampler):
    """
    Normal Mode Scanning class
    """

    def __init__(
        self,
        nms_data_file: Optional[str] = None,
        nms_harmonic_energy_step: Optional[float] = None,
        nms_energy_limits: Optional[Union[float, List[float]]] = None,
        nms_limit_of_coupling: Optional[int] = None,
        nms_limit_of_steps: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize Normal Mode Scanning class

        Parameters
        ----------

        nms_data_file: str, optional, default 'sample.db'
            Database file name to store the sampled systems with computed
            reference data.
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
            
        Returns
        -------
        object
            Normal Mode Scanning class object
        """
        
        # Initialize parent class with following parameter in kwargs:
        #config: Optional[Union[str, dict, object]] = None,
        #sample_directory: Optional[str] = None,
        #sample_data_file: Optional[str] = None,
        #sample_systems: Optional[Union[str, List[str], object]] = None,
        #sample_systems_format: Optional[Union[str, List[str]]] = None,
        #sample_calculator: Optional[Union[str, object]] = None,
        #sample_calculator_args: Optional[Dict[str, Any]] = None,
        #sample_properties: Optional[List[str]] = None,
        #sample_systems_optimize: Optional[bool] = None,
        #sample_systems_optimize_fmax: Optional[float] = None,
        super().__init__(**kwargs)
        
        #################################
        # # # Check NMS Class Input # # #
        #################################
        
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
        
        # Check sample data file
        if self.nms_data_file is None:
            self.nms_data_file = os.path.join(
                self.sample_directory, f'{self.sample_counter:d}_nms.db')
        elif not utils.is_string(self.nms_data_file):
            raise ValueError(
                f"Sample data file 'nms_data_file' must be a string " +
                f"of a valid file path but is of type " + 
                f"'{type(self.nms_data_file)}'.")
        
        # Sampler class label
        self.sample_tag = 'nmscan'
        
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
        print(self.sample_properties)
        self.nms_dataset = data.DataSet(
            self.nms_data_file,
            self.sample_unit_positions,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=True)
        
        return


    def get_info(self):
        
        return {
            'sample_directory': self.sample_directory,
            #'sample_data_file': self.sample_data_file,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'nms_data_file': self.nms_data_file,
            'nms_harmonic_energy_step': self.nms_harmonic_energy_step,
            'nms_energy_limits': self.nms_energy_limits,
            'nms_limit_of_coupling': self.nms_limit_of_coupling,
            'nms_limit_of_steps': self.nms_limit_of_steps,
        }


    #def run(
        #self,
        #nms_systems_idx: Optional[Union[int, List[int]]] = None,
    #):
        #"""
        #Perform Normal Mode Scanning on all sample systems or a
        #selection of them.

        #Parameters
        #----------

        #nms_systems_idx: (int, list(int)), optional, default None
            #Index or list of indices to run normal mode scanning only 
            #for the respective systems of the sample system list
        #"""
        
        ################################
        ## # # Check NMS Run Input # # #
        ################################
        
        ## Collect NMS sampling parameters
        #config_nms = {
            #f'{self.sample_counter}_{self.sample_tag}': 
                #self.get_info()
            #}
        
        ## Check sample system selection
        #nms_systems_selection = self.check_system_idx(nms_systems_idx)
        
        ## Update sampling parameters
        #config_nms['nms_systems_idx'] = nms_systems_idx
        
        ## Update configuration file with sampling parameters
        #if 'sampler_schedule' not in self.config:
            #self.config['sampler_schedule'] = {}
        #self.config['sampler_schedule'].update(config_nms)
        
        ## Increment sample counter
        #self.config['sample_counter'] = self.sample_counter
        #self.sample_counter += 1
        
        #########################################
        ## # # Perform Normal Mode Sampling # # #
        #########################################
        
        ## Iterate over systems
        #for system in self.sample_systems_atoms:
            
            ## Skip unselected system samples
            #if not nms_systems_selection:
                #continue
            
            ## If requested, perform structure optimization
            #if self.nms_systems_optimize:
                
                ## Assign ASE optimizer
                #ase_optimizer = optimize.BFGS

                ## Perform structure optimization
                #ase_optimizer(system).run(
                    #fmax=self.nms_systems_optimize_fmax)
                
            ## Start normal mode sampling
            #self.run_system(system)
            
    
    def run_system(self, system):
        """
        Perform Normal Mode Scanning on the sample system.
        """
        
        # Prepare system parameter
        Natoms = system.get_global_number_of_atoms()
        Nmodes = 3*Natoms
        
        # Compute initial state properties
        # TODO Special calculate property function to handle special properties
        # not supported by ASE such as, e.g., charge, hessian, etc.
        self.sample_calculator.calculate(system)
        init_properties = system._calc.results
        system_init_potential = init_properties['energy']
        
        # Add initial state properties to dataset
        self.nms_dataset.add_atoms(system, init_properties)
        
        # Perform numerical normal mode analysis
        ase_vibrations = vibrations.Vibrations(
            system,
            name=os.path.join(self.sample_directory, f"vib"))
        ase_vibrations.clean()
        ase_vibrations.run()
        ase_vibrations.summary()
        
        # (Trans. + Rot. + ) Vibrational frequencies in cm**-1
        system_frequencies = ase_vibrations.get_frequencies()
        
        # (Trans. + Rot. + ) Vibrational modes normalized to 1
        system_modes = np.array([
            ase_vibrations.get_mode(imode).reshape(Natoms, 3)
            /np.sqrt(np.sum(
                ase_vibrations.get_mode(imode).reshape(Natoms, 3)**2))
            for imode in range(Nmodes)])
        
        # Reduced mass per mode (in amu)
        system_redmass = np.array([
            1./np.sum(
                system_modes[imode]**2/system.get_masses().reshape(
                    Natoms, 1))
            for imode in range(Nmodes)])
        
        # Force constant per mode (in eV/Angstrom**2)
        system_forceconst = (
            4.0*np.pi**2*(np.abs(system_frequencies)*1.e2*units._c)**2
            *system_redmass*units._amu*units.J*1.e-20)
        
        # Compute and store equilibrium positions and center of mass,
        # moments of inertia and principle axis of inertia
        system_init_positions = system.get_positions()
        system_init_com = system.get_center_of_mass()
        
        # Compute and compare same quantities for displaced system
        system_com_shift = np.zeros(Nmodes, dtype=float)
        for imode, mode in enumerate(system_modes):
            
            # COM shift
            system.set_positions(system_init_positions + mode)
            system_displ_com = system.get_center_of_mass()
            system_com_shift[imode] = np.sqrt(np.sum(
                (system_init_com - system_displ_com)**2))
            
        # Vibrational modes are assumed with center of mass shifts smaller
        # than 1.e-3 Angstrom displacement
        system_vib_modes = system_com_shift < 1.e-3
        
        # Displacement factor for energy step (in eV)
        system_displfact = np.sqrt(
            3*self.nms_harmonic_energy_step/system_forceconst)
        
        # Iterate over number of normal mode combinations
        steps = np.arange(0, self.nms_limit_of_steps, 1) + 1
        vib_modes = np.where(system_vib_modes)[0]
        for icomp in range(1, self.nms_limit_of_coupling + 1):
            
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
                        for imode, modei in enumerate(imodes):
                            current_step_positions = (
                                system_init_positions
                                + system_displfact[modei]*system_modes[modei]
                                *current_step[imode])
                        system.set_positions(current_step_positions)
                        
                        # Compute observables and check potential threshold
                        try:
                            
                            self.sample_calculator.calculate(system)
                            current_properties = system._calc.results
                            current_potential = current_properties['energy']
                            #current_properties = (
                                #system._calc.calculate(properties=properties))
                            
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
                            self.nms_dataset.add_atoms(
                                system, current_properties)
                        
                        # Check energy threshold
                        if threshold_reached:
                            
                            # Return error if even initial step is to large
                            if istep == 0:
                                raise ValueError(
                                    f"Energy step size of " +
                                    f"{self.nms_harmonic_energy_step:.3f} "
                                    f"is too large!")
                                
                            # Check for next suitable step size index
                            istep += 1
                            for jstep in range(istep, Nsteps):
                                if np.any(
                                        np.array(all_steps[jstep]) 
                                        < step_size
                                    ):
                                    istep = jstep
                                    break
                            else:
                                istep = Nsteps
                                
                        else:
                            
                            # Increment step size index
                            istep += 1

                        # Check step size progress
                        if istep >= Nsteps:
                            done = True
              
        return