import sys
import pandas
import ctypes
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils
from .. import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['PyCharmm_Calculator']


class PyCharmm_Calculator:

    def __init__(
        self,
        model_calculator: Union[object, List[object]],
        # Total number of atoms
        num_atoms: int,
        # PhysNet atom indices
        ml_atom_indices: List[int],
        # PhysNet atom numbers
        ml_atom_numbers: List[int],
        # Fluctuating ML charges for ML-MM electrostatic interaction
        ml_fluctuating_charges: bool,
        # System atom charges (All atoms)
        ml_mm_atoms_charge: List[float],
        # Total charge of the system
        ml_total_charge: float,
        # Cutoff distance for ML/MM electrostatic interactions
        mlmm_rcut: float,
        # Cutoff width for ML/MM electrostatic interactions
        mlmm_width: float,
        # By default only energy and forces are calculated, maybe in the future dipoles
        implemented_properties: Optional[List[str]] = None,
        kehalf: float = 7.199822675975274,
        dtype=torch.float64,
        **kwargs
    ):

        # Initialize basic parameters
        self.num_atoms = num_atoms

        # Number of machine learning atoms
        self.ml_num_atoms = len(ml_atom_indices)

        # Number of MM atoms
        self.mm_num_atoms = self.num_atoms - self.ml_num_atoms

        # ML atom indices
        self.ml_indices =  torch.tensor(ml_atom_indices, dtype=torch.float64)

        # ML atom numbers
        self.ml_numbers = torch.tensor(ml_atom_numbers, dtype=torch.int64)

        # ML fluctuating charges
        self.ml_fluctuating_charges = ml_fluctuating_charges

        # ml mm atoms
        self.ml_mm_atoms_charge = torch.tensor(ml_mm_atoms_charge, dtype=torch.float64)

        # ML atom total charge
        self.ml_total_charge = torch.tensor(ml_total_charge, dtype=torch.float64)

        # ML atoms - atom indices pointing from MLMM position to ML position
        # 0, 1, 2 ..., ml_num_atoms: ML atom 1, 2, 3 ... ml_num_atoms + 1
        # ml_num_atoms + 1: MM atoms
        ml_idxp = np.full(self.num_atoms, -1)
        for ia, ai in enumerate(self.ml_indices):
            ml_idxp[ai] = ia
        self.ml_idxp = torch.tensor(ml_idxp, dtype=torch.int64)

        # Electrostatic interaction range
        self._mlmm_rcut = torch.tensor(mlmm_rcut, dtype=torch.float64)
        self._mlmm_rcut2 = torch.tensor(mlmm_rcut**2, dtype=torch.float64)
        self._mlmm_width = torch.tensor(mlmm_width, dtype=torch.float64)
        self.kehalf = kehalf
        # Set the dtype
        self.dtype = dtype


        # Settings of the model calculator

        # In case of model calculator is a list of models
        if utils.is_array_like(model_calculator):
            self.model_calculator = None
            self.model_calculator_list = model_calculator
            self.model_calculator_num = len(model_calculator)
            self.model_ensemble = True
        else:
            self.model_calculator = model_calculator
            self.model_calculator_list = None
            self.model_calculator_num = 1
            self.model_ensemble = False

        # Set implemented properties
        if implemented_properties is None:
            if self.model_ensemble:
                self.implemented_properties = (
                    self.model_calculator_list.model_properties)
            else:
                self.implemented_properties = (
                    self.model_calculator.model_properties)
        else:
            if utils.is_string(implemented_properties):
                self.implemented_properties = [implemented_properties]
            else:
                self.implemented_properties = implemented_properties

        # Check model properties and set evaluation mode
        if self.model_ensemble:
            for ic, calc in enumerate(self.model_calculator_list):
                for prop in self.implemented_properties:
                    if prop not in calc.model_properties:
                        raise SyntaxError(
                            f"Model calculator {ic:d} does not predict "
                            + f"property {prop:s}!\n"
                            + "Specify 'implemented_properties' with "
                            + "properties all model calculator support.")

        ##################################
        # # # Set Calculator Options # # #
        ##################################

        # Get model interaction cutoff and set evaluation mode
        if self.model_ensemble:
            self.interaction_cutoff = 0.0
            for calc in self.model_calculator_list:
                cutoff = calc.model_interaction_cutoff
                if self.interaction_cutoff < cutoff:
                    self.interaction_cutoff = cutoff
                calc.eval()
        else:
            self.interaction_cutoff = (
                self.model_calculator.model_interaction_cutoff)
            self.model_calculator.eval()

        # Initialize the non-bonded interaction calculator
        if self.ml_fluctuating_charges:
            self.non_bonded = Charmm_electrostatic(
                self._mlmm_rcut, self._mlmm_width, self.ml_idxp,
                self.ml_mm_atoms_charge, kehalf=self.kehalf)
        else:
            self.non_bonded = None

# Note: It needs to create a dictionary that contains the following values:
#         atoms_number = batch['atoms_number']
#         atomic_numbers = batch['atomic_numbers']
#         positions = batch['positions']
#         idx_i = batch['idx_i']
#         idx_j = batch['idx_j']
#         charge = batch['charge']
#         idx_seg = batch['atoms_seg']
#         pbc_offset = batch.get('pbc_offset')


    def calculate_charmm(self,Natom,Ntrans,Natim,x,y,z,dx,dy,dz,imattr,
                         Nmlp,Nmlmmp, idxi,idxj,idxu,idxv,idxp):
        '''

        This function matches the signature of the corresponding MLPOt in Pycharmm
        '''

        #Assing all positions
        if Ntrans:
            mlmm_R = torch.transpose(torch.tensor(
                [x[:Natim],y[:Natim],z[:Natim]],dtype=self.dtype).shape(3,Natom)
                                     ,0,1)
        else:
            mlmm_R = torch.transpose(torch.tensor(
                [x[:Natom],y[:Natom],z[:Natom]],dtype=self.dtype).shape(3,Natom)
                                     ,0,1)
        # Define indexes
        #Machine learning indexes
        ml_idxi = torch.tensor(idxi[:Nmlp],dtype=torch.int64).shape(Nmlp)
        ml_idxj = torch.tensor(idxj[:Nmlp],dtype=torch.int64).shape(Nmlp)
        # ML-MM indexes
        mlmm_idxi = torch.tensor(idxu[:Nmlmmp],dtype=torch.int64).shape(Nmlmmp)
        mlmm_idxk = torch.tensor(idxv[:Nmlmmp],dtype=torch.int64).shape(Nmlmmp)
        mlmm_idxk_p = torch.tensor(idxp[:Nmlmmp],dtype=torch.int64).shape(Nmlmmp)

        # Create batch for evaluating the model
        atoms_batch = {}
        atoms_batch['atoms_number'] = Natom
        atoms_batch['atomic_numbers'] = self.ml_numbers
        atoms_batch['positions'] = mlmm_R
        atoms_batch['idx_i'] = ml_idxi
        atoms_batch['idx_j'] = ml_idxj


        # Compute model properties
        results = {}
        if self.model_ensemble:
            # TODO Test!!!
            for ic, calc in enumerate(self.model_calculator_list):
                results[ic] = calc(atoms_batch)
            for prop in self.implemented_properties:
                prop_std = f"std_{prop:s}"
                results[prop_std], self.results[prop] = torch.std_mean(
                    torch.cat(
                        [
                            results[ic][prop]
                            for ic in range(self.model_calculator_num)
                        ],
                        dim=0),
                    dim=0)
        else:
            results = self.model_calculator(atoms_batch)

        # Calculate electrostatic energy
        if self.non_bonded is not None:
            mlmm_Eele =  self.non_bonded.non_bonded(
                mlmm_R, results['atomic_charges'], mlmm_idxi, mlmm_idxk, mlmm_idxk_p)
        else:
            mlmm_Eele = 0.0

        results['energy'] = results['energy'] + mlmm_Eele

        # Re-Calculate forces with the electrostatic energy
        gradient = torch.autograd.grad(
            torch.sum(results['energy']),
            mlmm_R,
            create_graph=True)[0]
        if gradient is not None:
            results['forces'] = -gradient
        else:
            results['forces'] = torch.zeros_like(mlmm_R)

        # Add unit conversion

        E = results['energy'].detach().numpy()
        F = results['forces'].detach().numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Reshaping forces
        for idxa in range(Natom):
            ii = 3*idxa
            dx[idxa] += F[ii]
            dy[idxa] += F[ii+1]
            dz[idxa] += F[ii+2]
        if Ntrans:
            for ii in range(Natom, Natim):
                idxa = imattr[ii]
                jj = 3*ii
                dx[idxa] += F[jj]
                dy[idxa] += F[jj+1]
                dz[idxa] += F[jj+2]

        return E


class Charmm_electrostatic:

    def __init__(self,mlmm_rcut,mlmm_width,ml_idxp, ml_mm_atoms_charge
                 ,switch_fn='CHARMM',kehalf=7.199822675975274):

        self.mlmm_rcut = mlmm_rcut
        self.ml_idxp = ml_idxp
        self.ml_mm_atoms_charge = ml_mm_atoms_charge
        # Initialize the class for cutoff
        switch_class = layers.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(self.mlmm_rcut, mlmm_width)
        self.kehalf = kehalf

    def calculate_mlmm_interatomic_distances(self,R,idxi,idxk,idxp):

        # Gather positions
        Ri = torch.gather(R,0,idxi.view(-1,1).repeat(1,3))
        Rk = torch.gather(R,0,idxk.view(-1,1).repeat(1,3))

        sum_distance = torch.sum((Ri-Rk)**2,dim=-1)
        idxr = torch.squeeze(torch.where(sum_distance < self.mlmm_rcut**2,sum_distance,
                                         torch.zeros_like(sum_distance)))

        p = torch.nn.ReLU(inplace=True)
        # Interacting atom pair distances
        Dik_temp = torch.gather(sum_distance,0,idxr) #Check shape of idxr
        Dik = torch.sqrt(p(Dik_temp))

        # Reduce the indexes to consider only interacting pairs

        idxi_r = torch.gather(idxi,0,idxr)
        idxp_r = torch.gather(idxp,0,idxr)

        return Dik,idxi_r,idxp_r

    def electrostatic_energy_per_atom_to_point_charge(self,Dik,Qai,Qak):
        ''' Calculate electrostatic interaction between QM atom charge and
            MM point charge based on shifted Coulomb potential scheme'''

        # Cutoff weighted reciprocal distance
        cutoff = self.switch_fn(Dik)
        rec_d = cutoff/Dik

        # Shifted Coulomb energy

        QQ = 2.0*self.kehalf*Qai*Qak
        Eele = QQ/Dik - QQ/self.mlmm_rcut*(2.0-Dik/self.mlmm_rcut)

        return torch.sum(cutoff*Eele)

    def non_bonded(self, mlmm_R, ml_Qa, mlmm_idxi, mlmm_idxk, mlmm_idxk_p):
        ''' Calculates the electrostatic interaction between ML atoms in
                    the center cell with all MM atoms in the non-bonded lists'''

        # Calculate ML(center) - MM(center,image) atom distances
        mlmm_Dik, mlmm_idxi_r, mlmm_idxp_r = \
            self.calculate_mlmm_interatomic_distances(
                mlmm_R, mlmm_idxi, mlmm_idxk, mlmm_idxk_p)

        mlmm_idxi_z = torch.gather(self.ml_idxp,0,mlmm_idxi_r)

        # Get ML and MM charges

        ml_Qai_r = torch.gather(ml_Qa,0,mlmm_idxi_z)
        ml_Qak_r = torch.gather(self.ml_mm_atoms_charge,0,mlmm_idxp_r)

        # Calculate electrostatic energy

        Eele = self.electrostatic_energy_per_atom_to_point_charge(
            mlmm_Dik,ml_Qai_r,ml_Qak_r)

        return Eele

















