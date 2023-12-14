# Adaptation of diffusion Monte Carlo code from https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main
import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import warnings
#Time importations
import time
from datetime import datetime

import numpy as np

import ase
from ase.io import read, write
from ase.io.trajectory import Trajectory

from .. import interface
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DMC']

class DMC:
    """

    This code adapt the diffusion Monte Carlo code from https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main

    **Original Message**
    DMC code for the calculation of zero-point energies using PhysNet based PESs on GPUs/CPUs. The calculation
    is performed in Cartesian coordinates.
    See e.g. American Journal of Physics 64, 633 (1996); https://doi.org/10.1119/1.18168 for DMC implementation
    and https://scholarblogs.emory.edu/bowman/diffusion-monte-carlo/ for Fortran 90 implementation.

    Parameters
    ----------


    """
    def __init__(self,natoms,nwalker,stepsize,nsteps,eqsteps,alpha,
                 atoms=None,atoms_charge=None,config=None,config_file=None,
                 model_checkpoint=None,implemented_properties=None,
                 use_neighbor_list=False,label='DMC',**kwargs):

        # Number of atoms
        self.natoms = natoms
        # Number of walkers for the DMC
        self.nwalker = nwalker
        # Step size for the DMC in imaginary time
        self.stepsize = stepsize
        # Number of steps for the DMC
        self.nsteps = nsteps
        # Number of equilibration steps for the DMC
        self.eqsteps = eqsteps
        # Alpha parameter for the DMC: Feed-back parameter, usually propotional to 1/stepsize"
        self.alpha = alpha

        #Unit conversion (Maybe move to the settings file)
        self.emass = 1822.88848

        # Atoms object
        if self.atoms is type(str):
            self.atoms = read(self.atoms)

        if self.atoms is None:
            raise ValueError('A geometry is required')

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
        self.dmc_trajectory = Trajectory(self.output, 'w', self.atoms, properties=['energy', 'forces'])

        # Seed is defined according to the time
        self.seed = np.random.seed(int(time.time()))

        # Informtation about the system

        self.natm = len(self.atoms)

        self.mass = np.sqrt(self.atoms.get_masses()*self.emass)
        self.nucl_charge = self.atoms.get_atomic_numbers()

        self.coord_min = self.atoms.get_positions()

        #I DON'T KNOW WHAT THIS IS!!
        Qref = 0.0
        max_batch = 6000

        atoms2 = self.atoms.copy()
        #Here we add a random displacement to the initial coordinates, I hardcoded the seed from the original code
        atoms2.rattle(stdev=0.1,seed=1453)

        self.initial_coord = atoms2.get_positions()

        self.logger = Logger_DMC(self.filename)

    def init_dmc(self):
        #Not sure this function is going to work,
        # There are some variables that are not defined in the original code. LIVS 14/12/2023

        #define stepsize
        self.deltax = np.sqrt(self.stepsize)/self.mass

        # psips_f keeps track of how many walkers are alive (psips_f[0]) and which ones (psips_f[1:], 1 for alive and 0 for dead)
        psips_f[:] = 1
        psips_f[0] = self.nwalker
        psips_f[self.nwalker + 1:] = 0

        # psips keeps track of atomic positions of all walkers
        # is initialized to some molecular geometry defined in the input xyz file
        psips[:, :, 0] = self.initial_coord[:]

        # reference energy (which is updated throughout the DMC simulation) is initialized to energy of v0, referenced to energy
        # of minimum geometry
        v_ref = v0
        v_ave = 0
        v_ref = v_ref - vmin
        self.logger.write_pot(psips_f[0], v_ref)

        return psips, psips_f, v_ave, v_ref

    def walk(self,psips, dx):
        """walk routine performs the diffusion process of the replicas by adding to the
           coordinates of the alive replicas sqrt(deltatau)rho, rho is random number
           from Gaussian distr
        """
        # print(psips.shape)
        dim = len(psips[0, :, 0])
        for i in range(dim):
            x = np.random.normal(size=(len(psips[:, 0, 0])))
            psips[:, i, 1] = psips[:, i, 0] + x * dx[np.ceil((i + 1) / 3.0) - 1]
            # print(psips[:,i-1,1])

        return psips

    def gbranch(self,refx, mass, symb, vmin, psips, psips_f, v_ref, v_tot, nalive):
        """The birth-death criteria for the ground state energy. Note that psips is of shape
           (3*nwalker, 3*natm) as only the progressed coordinates (i.e. psips[:,i,1]) are
           given to gbranch
        """

        birth_flag = 0
        error_checker = 0
        # print(psips.shape) #-> (3*nwalker, 3*natm)
        v_psip = get_batch_energy(psips[:nalive, :], nalive)  # predict energy of all alive walkers.

        # reference energy with respect to minimum energy.
        v_psip = v_psip - vmin

        # check for holes, i.e. check for energies that are lower than the one for the (global) min
        if np.any(v_psip < -1e-5):
            error_checker = 1
            idx_err = np.where(v_psip < -1e-5)
            record_error(refx, mass, symb, psips[idx_err, :], v_psip, idx_err)
            print("defective geometry is written to file")
            # kill defective walker idx_err + one as index 0 is counter of alive walkers
            psips_f[idx_err[0] + 1] = 0  # idx_err[0] as it is some stupid array...

        prob = np.exp((v_ref - v_psip) * stepsize)
        sigma = np.random.uniform(size=nalive)

        if np.any((1.0 - prob) > sigma):
            """test whether one of the walkers has to die given the probabilites
               and then set corresponding energies v_psip to zero as they
               are summed up later.
               geometries with high energies are more likely to die.
            """
            idx_die = np.array(np.where((1.0 - prob) > sigma)) + 1
            psips_f[idx_die] = 0
            v_psip[idx_die - 1] = 0.0

        v_tot = np.sum(v_psip)  # sum energies of walkers that are alive (i.e. fullfill conditions)

        if np.any(prob > 1):
            """give birth to new walkers given the probabilities and update psips, psips_f
               and v_tot accordingly.
            """
            idx_prob = np.array(np.where(prob > 1)).reshape(-1)

            for i in idx_prob:
                if error_checker == 0:

                    probtmp = prob[i] - 1.0
                    n_birth = int(probtmp)
                    sigma = np.random.uniform()

                    if (probtmp - n_birth) > sigma:
                        n_birth += 1
                    if n_birth > 2:
                        birth_flag += 1

                    while n_birth > 0:
                        nalive += 1
                        n_birth -= 1
                        psips[nalive - 1, :] = psips[i, :]
                        psips_f[nalive] = 1
                        v_tot = v_tot + v_psip[i]

                else:
                    if np.any(i == idx_err[0]):  # to make sure none of the defective geom are duplicated
                        pass
                    else:

                        probtmp = prob[i] - 1.0
                        n_birth = int(probtmp)
                        sigma = np.random.uniform()

                        if (probtmp - n_birth) > sigma:
                            n_birth += 1
                        if n_birth > 2:
                            birth_flag += 1

                        while n_birth > 0:
                            nalive += 1
                            n_birth -= 1
                            psips[nalive - 1, :] = psips[i, :]
                            psips_f[nalive] = 1
                            v_tot = v_tot + v_psip[i]

        error_checker = 0
        return psips, psips_f, v_tot, nalive







class Logger_DMC:

    def __init__(self,filename):

        self.filename = filename
        self.potfile = open(self.filename + ".pot",'w')
        self.logfile = open(self.filename + ".log",'w')
        self.errorfile = open("defective" + self.filename + ".xyz",'w')
        self.lastfile = open("configs" + self.filename + ".xyz",'w')


        # Units conversion
        self.auang = 0.5291772083
        self.aucm = 219474.6313710

    def log_begin(self,nwalker,nstep,eqstep,stepsize,alpha):
        """subroutine to write header of log file
           logging all job details
        """
        self.logfile.write("                  DMC for " + self.filename + "\n\n")
        self.logfile.write("DMC Simulation started at " + str(datetime.now()) + "\n")
        self.logfile.write("Number of random walkers: " + str(nwalker) + "\n")

        self.logfile.write("Number of total steps: " + str(nstep) + "\n")
        self.logfile.write("Number of steps before averaging: " + str(eqstep) + "\n")
        self.logfile.write("Stepsize: " + str(stepsize) + "\n")
        self.logfile.write("Alpha: " + str(alpha) + "\n\n")

    def log_end(self):
        """function to write footer of logfile
        """
        self.logfile.write("DMC Simulation terminated at " + str(datetime.now()) + "\n")
        self.logfile.write("DMC calculation terminated successfully\n")

    def write_error(self,refx,mass,symb,errq,v,idx):
        """
        subroutine to write xyz file of defective configurations

        Parameters
        ----------

        refx: array
            reference positions of atoms
        mass: array
            atomic masses
        symb: array
            atomic symbols
        errq: array
            error in positions
        v: array
            potential energy
        idx: array
            index of defective configurations
        """

        if len(idx[0]) == 1:
            natm = int(len(refx) / 3)
            errx = errq[0] * self.auang
            errx = errx.reshape(natm, 3)
            self.errorfile.write(str(int(natm)) + "\n")
            self.errorfile.write(str(v[idx[0]] * self.aucm) + "\n")
            for i in range(int(natm)):
                self.errorfile.write(
                    str(symb[i]) + "  " + str(errx[i, 0]) + "  " + str(errx[i, 1]) + "  " + str(errx[i, 2]) + "\n")

        else:
            natm = int(len(refx) / 3)
            errx = errq[0] * self.auang
            errx = errx.reshape(len(idx[0]), natm, 3)

            for j in range(len(errx)):
                self.errorfile.write(str(int(natm)) + "\n")
                self.errorfile.write(str(v[idx[0][j]] * self.aucm) + "\n")
                for i in range(int(natm)):
                    self.errorfile.write(str(symb[i]) + "  " + str(errx[j, i, 0]) + "  " + str(errx[j, i, 1]) + "  " + str(
                        errx[j, i, 2]) + "\n")

    def write_pot(self,psips_f,v_ref):
        """
        subroutine to write potential file
        Parameters
        ----------
        psips_f
        v_ref

        Returns
        -------

        """
        self.potfile.write("0  " + str(psips_f) + "  " + str(v_ref) + "  " + str(v_ref * self.aucm) + "\n")

    def write_log(self,v_ave):
        """
        subroutine to write average log file
        Parameters
        ----------
        v_ave

        Returns
        -------

        """
        self.logfile.write("AVERAGE ENERGY OF TRAJ   " + "   " + str(v_ave) + " hartree   " + str(v_ave * self.aucm) + " cm**-1\n")

