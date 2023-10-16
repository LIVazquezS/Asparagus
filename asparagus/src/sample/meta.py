import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import units

from ase.constraints import Hookean

from ase.io.trajectory import Trajectory

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from .. import data
from .. import settings
from .. import utils
from .. import sample

from ase.db import connect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MetaSampler']


class MetaSampler(sample.Sampler):
    """
    Meta(-Dynamic) Sampler class
    """
    
    def __init__(
        self,
        meta_cv: Optional[List[int]] = None,
        meta_gaussian_height: Optional[float] = None,
        meta_gaussian_widths: Optional[Union[float, List[float]]] = None,
        meta_gaussian_interval: Optional[int] = None,
        meta_hookean: Optional[List[Union[int, float]]] = None,
        meta_hookean_force_constant: Optional[float] = None,
        meta_temperature: Optional[float] = None,
        meta_time_step: Optional[float] = None,
        meta_simulation_time: Optional[float] = None,
        meta_save_interval: Optional[float] = None,
        meta_langevin_friction: Optional[float] = None,
        meta_initial_velocities: Optional[bool] = None,
        meta_initial_temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize Normal Mode Scanning class

        Parameters
        ----------

        meta_cv: list(list(int)), optional, default []
            List of sublists defining collective variables (CVs) / reaction 
            coordinates to add Gaussian potentials. The number of atom indices
            in the sublist defines either bonds (2), angles (3) or 
            dihedrals (4). Example [[1, 2], [1, 2, 3], [4, 1, 2, 3]]
        meta_gaussian_height: float, optional, default 0.05
            Potential energy height in eV of the Gaussian potential.
        meta_gaussian_widths: (float, list(floats)), optional, default 0.1
            Gaussian width for all CVs or a list of widths per CV that define
            the FWHM of Gaussian potential.
        meta_gaussian_interval: int, optional, default 10
            Step interval to add gaussian potential at current set of 
            collective variable.
        meta_hookean: list(list(int,float)), optional, default []
            It is always recommended for bond type collective variables to
            define a hookean constraint that limit the distance between two
            atoms. Otherwise gaussian potential could be only added to an ever
            increasing atoms bond distance. Hookean are defined by a list of
            sublists containing first two atom indices followed by one upper
            distance limit and, optionally, a Hookean force constant k (default
            is defined by meta_hookean_force_constant). 
            For example: [1, 2, 4.0] or [1, 2, 4.0, 5.0]
        meta_hookean_force_constant: float, optional, default 5.0
            Default Hookean force constant if not specifically defined in 
            Hookean constraint list meta_hookean.
        meta_temperature: float, optional, default 300
            Target temperature in Kelvin of the MD simulation controlled by a
            Langevin thermostat
        meta_time_step: float, optional, default 1.0 (1 fs)
            MD Simulation time step in fs
        meta_simulation_time: float, optional, default 1E5 (100 ps)
            Total MD Simulation time in fs
        meta_save_interval: int, optional, default 10
            MD Simulation step interval to store system properties of 
            the current frame to dataset.
        meta_langevin_friction: float, optional, default 0.1
            Langevin thermostat friction coefficient in Kelvin. The coefficient
            should be much higher than in classical MD simulations  (1E-2 to 
            1E-4) due to the fast heating of the systems due to the Gaussian 
            potentials.
        meta_initial_velocities: bool, optional, default False
            Instruction flag if initial atom velocities are assigned with
            respect to a Maxwell-Boltzmann distribution at temperature
            'md_initial_temperature'.
        meta_initial_temperature: float, optional, default 300
            Temperature for initial atom velocities according to a Maxwell-
            Boltzmann distribution.
        
        Returns
        -------
        object
            Meta(-Dynamics) Sampler class object
        """
        
        # Sampler class label
        self.sample_tag = 'meta'
        
        # Initialize parent class
        super().__init__(sample_tag=self.sample_tag, **kwargs)
        
        ##################################
        # # # Check Meta Class Input # # #
        ##################################
        
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
        
        # Get number of collective variables
        Ncv = len(self.meta_cv)
        
        # Check collective variables
        self.cv_type = []
        self.cv_type_dict = {
            2: 'bond',
            3: 'angle',
            4: 'dihedral',
            5: 'bondmix'}
        for icv, cv in enumerate(self.meta_cv):
            
            # Check cv data type
            if not utils.is_integer_array(cv):
                raise ValueError(
                    f"Collective variable number {icv:d} is not an integer " +
                    f"list but of type '{type(cv):s}'!")
            
            # Get cv type
            if self.cv_type_dict.get(len(cv)) is None:
                raise ValueError(
                    f"Collective variable number {icv:} is not of valid " +
                    f"length but of length '{len(cv):d}'!")
            else:
                self.cv_type.append(self.cv_type_dict.get(len(cv)))
        
        # Check meta sampling input format
        if utils.is_numeric(self.meta_gaussian_widths):
            self.meta_gaussian_widths = [self.meta_gaussian_widths]*Ncv
        elif Ncv > len(self.meta_gaussian_widths):
            raise ValueError(
                f"Unsufficient number of gaussian width defined " +
                f"({len(self.meta_gaussian_widths):d}) for the number " +
                f"of collective variables with {Ncv:d}")
        
        # Check Hookean constraints
        for ihk, hk in enumerate(self.meta_hookean):
            
            # Check Hookean list data type
            if not utils.is_numeric_array(hk):
                raise ValueError(
                    f"Hookean constraint number {ihk:d} is not a numeric " +
                    f"list but of type '{type(hk):s}'!")
            
            # Check Hookean constraint definition validity
            if len(hk)==3:
                # Add default Hookean force constant
                hk.append(self.meta_hookean_force_constant)
            elif not len(hk)==4:
                raise ValueError(
                    f"Hookean constraint number {ihk:d} is expected to be " +
                    f"length 3 or 4 but has a length of {len(hk):d}!")
            
            # Check atom definition type
            for ii, idx in enumerate(hk[:2]):
                if not utils.is_integer(idx):
                    raise ValueError(
                        f"Atom index {ii:d} in Hookean constraint number " +
                        f"{ihk:d} is not an integer but of type {type(idx)}!")

        # Define log file paths
        self.meta_simulation_log_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}_simulation.log')
        self.meta_gaussian_log_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}_gaussian.log')
        self.meta_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}.traj')
        
        # Check sample properties for energy and forces properties which are 
        # required for Meta sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')
        
        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################
        
        self.meta_dataset = data.DataSet(
            self.sample_data_file,
            self.sample_properties,
            self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

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
            'meta_cv': self.meta_cv,        
            'meta_gaussian_height': self.meta_gaussian_height,
            'meta_gaussian_widths': self.meta_gaussian_widths,
            'meta_gaussian_interval': self.meta_gaussian_interval,
            'meta_hookean': self.meta_hookean,
            'meta_hookean_force_constant': self.meta_hookean_force_constant,
            'meta_temperature': self.meta_temperature,
            'meta_time_step': self.meta_time_step,
            'meta_simulation_time': self.meta_simulation_time,
            'meta_save_interval': self.meta_save_interval,
            'meta_langevin_friction': self.meta_langevin_friction,
            'meta_initial_velocities': self.meta_initial_velocities,
            'meta_initial_temperature': self.meta_initial_temperature,
        }

    def run_system(self, system):
        """
        Perform Meta Sampling Simulation with the sample system.
        """
        
        # Initialize meta dynamic constraint
        meta_constraint = MetaConstraint(
            self.meta_cv,
            self.meta_gaussian_widths,
            self.meta_gaussian_height,
            self.meta_gaussian_log_file,
            self.meta_gaussian_interval
            )
        
        # Initialize Hookean constraint
        meta_hookean_constraint = []
        for hk in self.meta_hookean:
            meta_hookean_constraint.append(
                Hookean(hk[0], hk[1], hk[2], rt=hk[3]))
        
        # Set constraints to system
        system.set_constraint(
            [meta_constraint] + meta_hookean_constraint)

        # Set initial atom velocities if requested
        if self.meta_initial_velocities:
            MaxwellBoltzmannDistribution(
                system, 
                temperature_K=self.meta_initial_temperature)
        
        # Initialize MD simulation propagator
        meta_dyn = Langevin(
            system, 
            timestep=self.meta_time_step*units.fs,
            temperature_K=self.meta_temperature,
            friction=self.meta_langevin_friction,
            logfile=self.meta_simulation_log_file,
            loginterval=self.meta_save_interval)
        
        # Attach system properties saving function
        meta_dyn.attach(
            self.save_properties,
            interval=self.meta_save_interval,
            system=system)
        
        # Attach trajectory
        self.meta_trajectory = Trajectory(
            self.meta_trajectory_file, atoms=system, 
            mode='a', properties=self.sample_properties)
        meta_dyn.attach(
            self.write_trajectory, 
            interval=self.meta_save_interval,
            system=system)
        
        # Attach collective variables writer
        meta_cv_logger = MetaDynamicLogger(meta_constraint, system)
        meta_dyn.attach(
            meta_cv_logger.log, interval=self.meta_gaussian_interval)
        
        # Run MD simulation
        meta_simulation_step = round(
            self.meta_simulation_time/self.meta_time_step)
        meta_dyn.run(meta_simulation_step)
        

    def save_properties(self, system):
        """
        Save system properties
        """
        
        system_properties = self.get_properties(system)
        self.meta_dataset.add_atoms(system, system_properties)
        
        
    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """
        
        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.meta_trajectory.write(system_noconstraint)
        

class MetaConstraint:
    """
    Constraint class to perform Meta Dynamics simulations by adding
    artificial Gaussian potentials.
    """
    
    def __init__(
        self, 
        cv, 
        widths, 
        height, 
        logfile, 
        logwrite,
    ):
        """
        Forces atoms of cluster to stay close to the center.
        
        cv : list
            List of collective variables - e.g.:
                [[1,2], [2,4], [1,2,3], [2,3,4,1]]
            [a,b] bond variable between atom a and b
            [a,b,c] angle variable between angle a->b->c
            [a,b,c,d] dihedral angle variable between angle a->b->c->d
        widths : array
           Width if Gaussian distribution for cv i
        heights : array
           Maximum height of the artificial Gaussian potential
        """
        
        self.cv = cv
        self.Ncv = len(cv)
        self.widths = np.asarray(widths)
        self.height = height
        self.logfile = logfile
        self.logwrite = logwrite
        
        self.cv_list = np.array([], dtype=float)
        self.last_cv = None
        self.ilog = 0
        
        if self.Ncv != len(self.widths):
            raise IOError(
                'Please provide an array of width values of them' +
                'same lengths as the number of collective variables cv.')
        
        self.removed_dof = 0
                
    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        dct = {'name': 'MetaConstraints'}
        dct['kwargs'] = {'cv': self.cv,
                         'widths': self.widths,
                         'height': self.height}
        #dct['kwargs'] = {}
        return dct
    
    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass
    
    def adjust_forces(self, atoms, forces):
        """Returns the Forces related to artificial Gaussian potential"""
        
        # Get collective variable i and partial derivative dcvdR
        cv = np.zeros(self.Ncv, dtype=float)
        dcvdR = np.zeros([self.Ncv, *forces.shape], dtype=float)
        for icv, cvi in enumerate(self.cv):
            # Bond length
            if len(cvi)==2:
                cv[icv] = np.linalg.norm(
                    atoms.positions[cvi[0]] - atoms.positions[cvi[1]])
                dcvdR[icv, cvi[0]] = (
                    (atoms.positions[cvi[1]] - atoms.positions[cvi[0]])/cv[icv])
                dcvdR[icv, cvi[1]] = (
                    (atoms.positions[cvi[0]] - atoms.positions[cvi[1]])/cv[icv])
            # Angle
            elif len(cvi)==3:
                # Connection vectors
                ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                # Vector distances
                d2ab = np.sum(ab**2)
                dab = np.sqrt(d2ab)
                d2cb = np.sum(cb**2)
                dcb = np.sqrt(d2cb)
                # Vector angles
                cabbc = np.sum(ab*cb)/(dab*dcb)
                cabbc = np.clip(cabbc, -1.0, 1.0)
                sabbc = np.sqrt(1.0 - cabbc**2)
                sabbc = np.clip(sabbc, 1.e-8, None)
                tabbc = np.arccos(cabbc)
                # Add to current collective variable
                cv[icv] = tabbc
                # Compute gradients dcvdRi
                a11 = sabbc*cabbc/d2ab
                a12 = -sabbc/(dab*dcb)
                a22 = sabbc*cabbc/d2cb
                fab = a11*ab + a12*cb
                fcb = a22*cb + a12*ab
                dcvdR[icv, cvi[0]] = -fab
                dcvdR[icv, cvi[1]] = fab + fcb
                dcvdR[icv, cvi[2]] = -fcb
            # Dihedral angle
            elif len(cvi)==4:
                ## Connection vectors
                #ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                #cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                #dc = atoms.positions[cvi[3]] - atoms.positions[cvi[2]]
                #bc = -cb
                ## Plane vectors
                #aa = np.cross(ab, bc)
                #bb = np.cross(dc, bc)
                ## Vector distances
                #d2aa = np.sum(aa**2)
                #d2bb = np.sum(bb**2)
                #d2bc = np.sum(bc**2)
                #dbc = np.sqrt(d2bc)
                #dinvab = 1./np.sqrt(d2aa*d2bb)
                ## Vector angles
                #cabcd = dinvab*np.sum(aa*bb)
                #cabcd = np.clip(cabcd, -1.0, 1.0)
                #sabcd = dbc*dinvab*np.sum(aa*dc)
                ## Dihedral multiplicity scalars 
                ## (multiplicity: number of minima in dihedral 0° to 360°)
                ## Here: multiplicity equal 1
                #p, dfab, ddfab = 1.0, 0.0, 0.0
                #for ii in range(1):
                    #ddfab = p*cabcd - dfab*sabcd
                    #dfab = p*sabcd + dfab*cabcd
                    #p = ddfab
                
                
                
                
                raise NotImplementedError
            # Reaction coordinate
            elif len(cvi)==5:
                if cvi[0]=='-':
                    a = np.linalg.norm(
                        atoms.positions[cvi[1]] - atoms.positions[cvi[2]])
                    b = np.linalg.norm(
                        atoms.positions[cvi[3]] - atoms.positions[cvi[4]])
                    cv[icv] = a - b
                    dcvdR[icv, cvi[1]] = (
                        (atoms.positions[cvi[2]] - atoms.positions[cvi[1]])/a)
                    dcvdR[icv, cvi[2]] = (
                        (atoms.positions[cvi[1]] - atoms.positions[cvi[2]])/a)
                    dcvdR[icv, cvi[3]] = (
                        (atoms.positions[cvi[3]] - atoms.positions[cvi[4]])/b)
                    dcvdR[icv, cvi[4]] = (
                        (atoms.positions[cvi[4]] - atoms.positions[cvi[3]])/b)
                else:
                    raise NotImplementedError
            else:
                raise IOError('Check cv list of Metadynamic constraint!')
        
        # Put to last cv
        self.last_cv = cv.copy()
        
        # If CV list is empty return zero forces
        # (Important to put it after last_cv update for the add_to_cv function)
        if self.cv_list.shape[0]==0:
            return np.zeros_like(forces)
        
        # Compute Gaussian exponents
        #exponents = np.sum(
            #np.divide(
                #np.square(
                    #np.subtract(
                        #self.cv_list,
                        ##(Nlist, Ncv)
                        #np.expand_dims(cv, axis=0)
                        ##(1, Ncv)
                        #)
                    ##(Nlist, Ncv)
                    #),
                ##(Nlist, Ncv)
                #np.expand_dims(
                    #np.multiply(
                        #2.0,
                        #np.square(
                            #self.widths
                            ##(Ncv)
                            #)
                        ##(Ncv)
                        #),
                    ##(Ncv)
                    #axis=0
                    #)
                ##(1, Ncv)
                #),
            ##(Nlist, Ncv)
            #axis=1
            #)
        ##(Nlist)
        exponents = np.sum(
            (self.cv_list - np.expand_dims(cv, axis=0))**2
            /np.expand_dims(2.0*self.widths**2, axis=0),
            axis=1)
        #(Nlist)
        
        # Compute Gaussians
        gaussians = -1.0*self.height*np.exp(-exponents)
        #(Nlist)
        
        # Compute partial derivative d exponent d cv
        #dexpdcv = np.divide(
            #np.subtract(
                #self.cv_list,
                ##(Nlist, Ncv)
                #np.expand_dims(
                    #cv,
                    ##(Ncv)
                    #axis=0
                    #)
                ##(1, Ncv)
                #),
            ##(Nlist, Ncv)
            #np.expand_dims(
                #np.square(
                    #self.widths
                    ##(Ncv)
                    #),
                ##(Ncv)
                #axis=0
                #)
            ##(1, Ncv)
            #)
        ##(Nlist, Ncv)
        dexpdcv = (
            (self.cv_list -  np.expand_dims(cv, axis=0))
            /np.expand_dims(self.widths**2, axis=0))
        #(Nlist, Ncv)
        
        # Add up gradient with respective to cv
        #dgausdcv = np.sum(
            #np.multiply(
                #np.expand_dims(
                    #gaussians,
                    ##(Nlist)
                    #axis=1
                    #),
                ##(Nlist, 1)
                #dexpdcv
                #),
            ##(Nlist, Ncv)
            #axis=0
            #)
        ##(Ncv)
        dgausdcv = np.sum(np.expand_dims(gaussians, axis=1)*dexpdcv, axis=0)
        #(Ncv)
        
        # Compute gradient with respect to Cartesian
        #gradient = np.sum(
            #np.multiply(
                #np.expand_dims(
                    #dgausdcv,
                    ##(Ncv)
                    #axis=(1,2)
                    #),
                ##(Ncv, 1, 1)
                #dcvdR
                ##(Ncv, Natoms, Ncart)
                #),
            #axis=0
            #)
        ##(Natoms, Ncart)
        gradient = np.sum(np.expand_dims(dgausdcv, axis=(1,2))*dcvdR, axis=0)
        #(Natoms, Ncart)
        
        #dvec = (atoms.positions[1] - atoms.positions[0])/cv[0]
        #dforce = -gradient*dvec.reshape(1, -1)
        ##dforce = np.sqrt(np.sum(dforce**2, axis=-1))
        ##dforce = dforce.reshape(-1, 1)*dvec.reshape(1, -1)
        ##dforce -= (dforce[0] + dforce[1])/2.
        #print(dforce)
        ##print(np.sqrt(np.sum(dforce**2)))
        #atoms_num = atoms.copy()
        #atoms_num.set_distance(0, 1, cv[0] - 0.005)
        #em = self.adjust_potential_energy(atoms_num)
        #atoms_num.set_distance(0, 1, cv[0] + 0.005)
        #ep = self.adjust_potential_energy(atoms_num)
        #print((ep - em)/.01)
        
        forces -= gradient
        
    def adjust_potential_energy(self, atoms):
        """Returns the artificial Gaussian potential"""
        
        # Get collective variable i
        cv = np.zeros(self.Ncv, dtype=float)
        for icv, cvi in enumerate(self.cv):
            
            # Bond length
            if len(cvi)==2:
                cv[icv] = np.linalg.norm(
                    atoms.positions[cvi[0]] - atoms.positions[cvi[1]])
            # Angle
            elif len(cvi)==3:
                # Connection vectors
                ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                # Vector distances
                d2ab = np.sum(ab**2)
                dab = np.sqrt(d2ab)
                d2cb = np.sum(cb**2)
                dcb = np.sqrt(d2cb)
                # Vector angles
                cabbc = np.sum(ab*cb)/(dab*dcb)
                cabbc = np.clip(cabbc, -1.0, 1.0)
                tabbc = np.arccos(cabbc)
                # Add to current collective variable
                cv[icv] = tabbc
            # Dihedral angle
            elif len(cvi)==4:
                raise NotImplementedError
            # Special
            elif len(cvi)==5:
                if cvi[0]=='-':
                    a = np.linalg.norm(
                        atoms.positions[cvi[1]] - atoms.positions[cvi[2]])
                    b = np.linalg.norm(
                        atoms.positions[cvi[3]] - atoms.positions[cvi[4]])
                    cv[icv] = a - b
                else:
                    raise NotImplementedError
            else:
                raise IOError('Check cv list of Metadynamic constraint!')
        
        # Update to last cv
        self.last_cv = cv.copy()
        
        # If CV list is empty return zero potential
        # (Important to put it after last_cv update for the add_to_cv function)
        if self.cv_list.shape[0]==0:
            return 0.0
        
        # Compute Gaussian exponents
        exponents = np.sum(
            (self.cv_list - np.expand_dims(cv, axis=0))**2
            /np.expand_dims(2.0*self.widths**2, axis=0),
            axis=1)
        #(Nlist)
        
        # Compute gaussians
        gaussians = self.height*np.exp(-exponents)
        #(Nlist)
        
        # Add up potential
        potential = np.sum(gaussians)
        
        return potential
    
    def add_to_cv(self, atoms):
        
        # Get collective variable i
        if self.last_cv is None:
            return
        else:
            cv = self.last_cv
        
        # Add cv to list
        if self.cv_list.shape[0]==0:
            self.cv_list = np.array([cv], dtype=float)
        else:
            self.cv_list = np.append(self.cv_list, [cv], axis=0)
        
        # Write cv list but just every logwrite time to avoid repeated writing 
        # of large files
        if not self.ilog%self.logwrite:
            np.savetxt(self.logfile, self.cv_list)
        self.ilog += 1
        
    def get_indices(self):
        raise NotImplementedError
        
    def index_shuffle(self, atoms, ind):
        raise NotImplementedError

    def __repr__(self):
        return (
            'Metadynamics constraint') 



class MetaDynamicLogger:
    """ 
    Store defined amount of images of a trajectory and save the images 
    within a defined time window around a point when an action happened.
    """
    
    def __init__(self, metadynamic, system):
        """
        Parameters:
        
        metadynamic: object
            Metadynamic constraint class
        """
        
        # Allocate parameter
        self.metadynamic = metadynamic
        self.system = system
        
    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        
    def log(self, system=None, **kwargs):
        """
        Execute 
        """
        
        if system is None:
            system = self.system
        self.metadynamic.add_to_cv(system)
        
        return
    
