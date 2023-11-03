# MLpot: Custom 
# Copyright (C) 2018 Josh Buckner

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" A class to set PhysNet model to run during an energy calculation
"""

import ctypes
import pandas
import pycharmm

import numpy as np
from asparagus import Asparagus


class MLpot():
    """
    Custom Machine Learning potential
    """
    
    def __init__(
        self,
        # ML atom number
        Z,
        #Config file
        config,
        # ML atom selection
        ml_selection,
        # Fluctuating ML atom charges for ML/MM electrostatic interaction
        ml_fq=True,
        # Additional model keyword arguments
        **kwargs):
        
        # ML&MM - atom number
        self.Natoms = pycharmm.coor.get_natom()
        
        # ML atoms - atom indices
        ml_indices = ml_selection.get_atom_indexes()
        ml_argsort = np.argsort(np.array(ml_indices, dtype=int))
        self.ml_indices = np.array(ml_indices, dtype=int)[ml_argsort]
        
        # ML atoms - atom number
        self.ml_Natoms = len(ml_indices)
        
        # ML - Delete bonds, angles, dihedrals, improper - if active
        pycharmm.psf.delete_bonds(ml_selection, ml_selection)
        pycharmm.psf.delete_angles(ml_selection, ml_selection)
        pycharmm.psf.delete_dihedrals(ml_selection, ml_selection)
        pycharmm.psf.delete_impropers(ml_selection, ml_selection)
        
        # ML&MM - psf charges - set ML charges zero when charges and ML-MM 
        # interaction are handled by the ML potential
        self.ml_fq = ml_fq
        self.mlmm_charges = np.array(pycharmm.param.get_charge())
        if ml_fq:
            self.mlmm_charges[self.ml_indices] = 0.0
            _ = pycharmm.psf.set_charge(self.mlmm_charges)
        
        # ML - set non-bond exclusion list for ML atom pairs
        self.ml_iblo = np.zeros(self.Natoms, dtype=int)
        self.ml_inb = []
        for ii, idx in enumerate(ml_indices):
            self.ml_iblo[idx:] += self.ml_Natoms - ii - 1
            for jdx in self.ml_indices[(ii + 1):]:
                self.ml_inb.append(jdx + 1) # + 1 as CHARMM start at index 1
        self.ml_nnb = len(self.ml_inb)
        
        pycharmm.psf.set_iblo_inb(self.ml_iblo, self.ml_inb)
        pycharmm.nbonds.update_bnbnd() # Already executed in set_iblo_inb()
        pycharmm.image.update_bimag()
        
        ###################################################
        # START - Potential model dependent part
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        #VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV

        # ML atoms - atomic numbers
        if Z is not None:
            ml_Z = np.array(kwargs["Z"], dtype=int)[ml_argsort]
        else:
            raise IOError("ML atom number 'Z' for the selected atoms is mandatory!")
        # Respective config file for PhysNet architecture
        if config is not None:
            config = config
        else:
            raise IOError("Asparagus config file is mandatory!")
        # Checkpoint file to restore the model 
        if "checkpoint" in kwargs.keys():
            checkpoint = kwargs["checkpoint"]
        else:
            print('WARNING: No checkpoint file is provided! By Default, Asparagus will use the best model.')
            checkpoint = None
        # System charge
        if "charge" in kwargs.keys():
            ml_charge = kwargs["charge"]
        else:
            ml_charge = 0
        # ML/MM electrostatic interaction max cutoff
        if "ctofnb" in kwargs.keys():
            ctofnb = kwargs["ctofnb"]
        else:
            ctofnb = None
        # ML/MM electrostatic interaction min cutoff
        if "ctonnb" in kwargs.keys():
            ctonnb = kwargs["ctonnb"]
        else:
            ctonnb = None


        self.model = Asparagus(
            config=config,
        )

        # # Read config file or dictionary
        # if isinstance(config, str):
        #
        #     with open(config, 'r') as f:
        #         config_lines = f.readlines()
        #
        #     # Get PhysNet parameter
        #     custom_params = {}
        #     for cline in config_lines:
        #
        #         # Read line
        #         cpar = cline.split("=")
        #         if len(cpar)!=2:
        #             print("Incorrect config line '{:s}' - skipped!". format(cline))
        #             print("Expect '--key item' format.")
        #             continue
        #         else:
        #             (key, item) = cpar
        #
        #         # Check key format
        #         if "--" == key[:2]:
        #             key = key[2:]
        #         else:
        #             print("Incorrect key format '{:s}' - skipped!". format(key))
        #             print("Expect '--key item' format.")
        #             continue
        #
        #         # Check key duplicity:
        #         if key in custom_params.keys():
        #             print("Multiple definitions of '{:s}' - skipped!". format(key))
        #             continue
        #
        #         # Append custom parameter
        #         custom_params[key] = item
        #
        # else:
        #
        #     custom_params = config
        #
        # # Update PhysNet parameters
        # self.params.update(custom_params)

            
        # ML/MM interaction range
        if ctofnb is None:
            ctofnb = pycharmm.nbonds.get_ctofnb()
        else:
            ctofnb = float(ctofnb)
        if ctonnb is None:
            ctonnb = pycharmm.nbonds.get_ctonnb()
        else:
            ctonnb = float(ctonnb)
        

        
        # Initialize PhysNet model
        self.calculator = self.model.get_pycharmm_calculator(
            self.Natoms,
            self.ml_indices,
            ml_Z,
            self.ml_fq,
            self.mlmm_charges,
            ml_charge,
            ctofnb,
            ctofnb - ctonnb,
            implemented_properties=['energy', 'forces','atomic_charges'],
            checkpoint=checkpoint,
        )

        
        # Initialize custom energy function
        self.func_type = ctypes.CFUNCTYPE(
            ctypes.c_double,                    # user energy - E(user)
            ctypes.c_int,                       # Atom number central cell - Natom
            ctypes.c_int,                       # Number of central and image cells - Ntrans
            ctypes.c_int,                       # Atom number central + image cells - Natim
            ctypes.POINTER(ctypes.c_double),    # x: Atom position x
            ctypes.POINTER(ctypes.c_double),    # y: Atom position y
            ctypes.POINTER(ctypes.c_double),    # z: Atom position y
            ctypes.POINTER(ctypes.c_double),    # dx: Atom potential derivative dE/dx
            ctypes.POINTER(ctypes.c_double),    # dy: Atom potential derivative dE/dy
            ctypes.POINTER(ctypes.c_double),    # dz: Atom potential derivative dE/dz
            ctypes.POINTER(ctypes.c_int),       # IMATTR: Corresponding central atom index of image atom
            ctypes.c_int,                       # Nmlp: Number of ML-ML atom pairs
            ctypes.c_int,                       # Nmlmmp: Number of ML-MM atom pairs
            ctypes.POINTER(ctypes.c_int),       # idxi: ML-ML atom pair (atom i)
            ctypes.POINTER(ctypes.c_int),       # idxj: ML-ML atom pair (atom j)
            ctypes.POINTER(ctypes.c_int),       # idxu: ML-MM atom pair (ML atom u)
            ctypes.POINTER(ctypes.c_int),       # idxv: ML-MM atom pair (MM atom v)
            ctypes.POINTER(ctypes.c_int)        # idxp: Image to central MM atom pointer)
            )
        
        self.energy_func = self.func_type(self.calculator.calculate_charmm)
        
        #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        #||||||||||||||||||||||||||||||||||||||||||||||||||
        # END - Potential model dependent part
        ###################################################
        
        pycharmm.lib.charmm.mlpot_set_func(self.energy_func)
        
        Nml = self.ml_Natoms
        mlidx = (ctypes.c_int * Nml)(*(self.ml_indices + 1))
        mlidz = (ctypes.c_int * Nml)(*ml_Z)
        Nml = (ctypes.c_int * 1)(Nml)
        pycharmm.lib.charmm.mlpot_set_properties(
            Nml, mlidx, mlidz)
        
        self.is_set = True
        
    def __del__(self):
        """class destructor
        """
        self.unset_mlpot()

    def unset_mlpot(self):
        """just store the function and do not run it during energy calcs
        """
        pycharmm.lib.charmm.mlpot_unset()
        self.is_set = False

    
