import sys
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

CHARMM_calculator_units = {
    'positions':        'Ang',
    'energy':           'kcal/mol',
    'forces':           'kcal/mol/Ang',
    'hessian':          'kcal/mol/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'e*Ang',
    }


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
        mlmm_atoms_charge: List[float],
        # Total charge of the system
        ml_total_charge: float,
        # Cutoff distance for ML/MM electrostatic interactions
        mlmm_rcut: float,
        # Cutoff width for ML/MM electrostatic interactions
        mlmm_width: float,
        # By default only energy and forces are calculated, maybe dipoles in
        # the future
        implemented_properties: Optional[List[str]] = None,
        dtype=torch.float64,
        **kwargs
    ):

        # Set the dtype
        self.dtype = dtype

        ################################
        # # # Set PyCHARMM Options # # #
        ################################

        # Initialize basic parameters
        self.num_atoms = num_atoms

        # Number of machine learning atoms
        self.ml_num_atoms = len(ml_atom_indices)

        # Number of MM atoms
        self.mm_num_atoms = self.num_atoms - self.ml_num_atoms

        # ML atom indices
        self.ml_indices = torch.tensor(ml_atom_indices, dtype=self.dtype)

        # ML atom numbers
        self.ml_numbers = torch.tensor(ml_atom_numbers, dtype=self.dtype)

        # ML fluctuating charges
        self.ml_fluctuating_charges = ml_fluctuating_charges

        # MM and ML atom charges
        self.mlmm_atoms_charge = torch.tensor(
            mlmm_atoms_charge, dtype=self.dtype)

        # ML atom total charge
        self.ml_total_charge = torch.tensor(
            ml_total_charge, dtype=self.dtype)

        # ML atoms - atom indices pointing from MLMM position to ML position
        # 0, 1, 2 ..., ml_num_atoms: ML atom 1, 2, 3 ... ml_num_atoms + 1
        # ml_num_atoms + 1: MM atoms
        ml_idxp = np.full(self.num_atoms, -1)
        for ia, ai in enumerate(self.ml_indices):
            ml_idxp[ai] = ia
        self.ml_idxp = torch.tensor(ml_idxp, dtype=torch.int64)

        # Non-bonding interaction range
        self.mlmm_rcut = torch.tensor(mlmm_rcut, dtype=self.dtype)
        self.mlmm_rcut2 = torch.tensor(mlmm_rcut**2, dtype=self.dtype)
        self.mlmm_width = torch.tensor(mlmm_width, dtype=self.dtype)

        ################################
        # # # Set Model Calculator # # #
        ################################

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

        #############################
        # # # Set ML/MM Options # # #
        #############################

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

        # Check cutoff of CHARMM and the ML model
        self.max_rcut = np.max([self.interaction_cutoff, self.mlmm_rcut])

        # Get property unit conversions from model units to CHARMM units
        self.model_unit_properties = (
            self.model_calculator.model_unit_properties)

        self.charmm_unit_properties = {}
        self.model2charmm_unit_conversion = {}

        # Positions unit conversion
        conversion, _ = utils.check_units(
            CHARMM_calculator_units['positions'],
            self.model_unit_properties['positions'])
        self.charmm_unit_properties['positions'] = conversion
        self.model2charmm_unit_conversion['positions'] = conversion

        # Implemented property units conversion
        for prop in self.implemented_properties:
            conversion, _ = utils.check_units(
                CHARMM_calculator_units[prop],
                self.model_unit_properties[prop])
            self.charmm_unit_properties[prop] = conversion
            self.model2charmm_unit_conversion[prop] = conversion

        # Initialize the non-bonded interaction calculator
        if self.ml_fluctuating_charges:

            # Convert 1/(2*4*pi*epsilon) from e**2/eV/Ang to CHARMM units
            kehalf_ase = 7.199822675975274
            conversion, _ = utils.check_units(
                self.model_unit_properties.get('energy'))
            self.kehalf = torch.tensor(
                [kehalf_ase*1.**2/conversion/1.],
                dtype=self.dtype)

            self.electrostatics_calc = Electrostatic_shift(
                self.mlmm_rcut,
                self.mlmm_width,
                self.max_rcut,
                self.ml_idxp,
                self.mlmm_atoms_charge,
                kehalf=self.kehalf)

        else:

            self.electrostatics_calc = None

# Note: It needs to create a dictionary that contains the following values:
#         atoms_number = batch['atoms_number']
#         atomic_numbers = batch['atomic_numbers']
#         positions = batch['positions']
#         idx_i = batch['idx_i']
#         idx_j = batch['idx_j']
#         charge = batch['charge']
#         idx_seg = batch['atoms_seg']
#         pbc_offset = batch.get('pbc_offset')
    def calculate_charmm(
        self,
        Natom: int,
        Ntrans: int,
        Natim: int,
        x: List[float],
        y: List[float],
        z: List[float],
        dx: List[float],
        dy: List[float],
        dz: List[float],
        imattr: List[int],
        Nmlp: int,
        Nmlmmp: int,
        idxi: List[int],
        idxj: List[int],
        idxu: List[int],
        idxv: List[int],
        idxp: List[int],
    ) -> float:
        """
        This function matches the signature of the corresponding MLPot class in
        PyCHARMM.
        """

        # Assign all positions
        if Ntrans:
            mlmm_R = torch.transpose(
                torch.tensor(
                    [x[:Natim], y[:Natim], z[:Natim]], dtype=self.dtype
                ).shape(3, Natom),
                0, 1)
        else:
            mlmm_R = torch.transpose(
                torch.tensor(
                    [x[:Natom], y[:Natom], z[:Natom]], dtype=self.dtype
                ).shape(3, Natom),
                0, 1)

        # Assign indices
        ml_idxi = torch.tensor(idxi[:Nmlp], dtype=torch.int64).shape(Nmlp)
        ml_idxj = torch.tensor(idxj[:Nmlp], dtype=torch.int64).shape(Nmlp)
        mlmm_idxi = torch.tensor(
            idxu[:Nmlmmp], dtype=torch.int64).shape(Nmlmmp)
        mlmm_idxk = torch.tensor(
            idxv[:Nmlmmp], dtype=torch.int64).shape(Nmlmmp)
        mlmm_idxk_p = torch.tensor(
            idxp[:Nmlmmp], dtype=torch.int64).shape(Nmlmmp)
        atom_seg = torch.zeros(Natom, dtype=torch.int64)

        # Create batch for evaluating the model
        atoms_batch = {}
        atoms_batch['atoms_number'] = Natom
        atoms_batch['atomic_numbers'] = self.ml_numbers
        atoms_batch['positions'] = mlmm_R
        atoms_batch['idx_i'] = ml_idxi
        atoms_batch['idx_j'] = ml_idxj
        atoms_batch['pbc_offset'] = None
        atoms_batch['atom_seg'] = atom_seg

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

        # Unit conversion
        self.results = {}
        for prop in self.implemented_properties:
            self.results[prop] = (
                results[prop]*self.charmm_unit_properties[prop])

        # Calculate electrostatic energy and force contribution
        if self.electrostatics_calc is not None:
            mlmm_Eele = self.electrostatics_calc.run(
                mlmm_R,
                self.results['atomic_charges'],
                mlmm_idxi,
                mlmm_idxk,
                mlmm_idxk_p)
            mlmm_gradient = torch.autograd.grad(
                torch.sum(mlmm_Eele),
                mlmm_R,
                create_graph=True)[0]

            # Add electrostatic interaction potential to ML energy
            self.results['energy'] = self.results['energy'] + mlmm_Eele

            # Add electrostatic interaction forces to ML and MM forces
            if mlmm_gradient is not None:
                self.results['forces'] -= mlmm_gradient

        ## Re-Calculate forces with the electrostatic energy
        #gradient = torch.autograd.grad(
            #torch.sum(self.results['energy']),
            #mlmm_R,
            #create_graph=True)[0]
        #if gradient is not None:
            #self.results['forces'] = -gradient

        # Apply unit conversion
        self.results['energy'] *= self.charmm_unit_properties['energy']
        self.results['forces'] *= self.charmm_unit_properties['forces']

        # Apply dtype conversion
        E = self.results['energy'].detach().numpy()
        F = self.results['forces'].detach().numpy().ctypes.data_as(
            ctypes.POINTER(ctypes.c_float))

        # Add forces to CHARMM derivative arrays
        # (Check addition or substration)
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


class Electrostatic_shift:

    def __init__(
        self,
        mlmm_rcut: torch.Tensor[float],
        mlmm_width: torch.Tensor[float],
        max_rcut: torch.Tensor[float],
        ml_idxp: torch.Tensor[int],
        mlmm_atoms_charge: torch.Tensor[float],
        kehalf: torch.Tensor[float],
        switch_fn='CHARMM',
    ):

        self.mlmm_rcut = mlmm_rcut
        self.ml_idxp = ml_idxp
        self.mlmm_atoms_charge = mlmm_atoms_charge
        self.max_rcut2 = max_rcut**2

        # Initialize the class for cutoff
        switch_class = layers.get_cutoff_fn(switch_fn)
        self.switch_fn = switch_class(mlmm_rcut, mlmm_width)
        self.kehalf = kehalf

    def calculate_mlmm_interatomic_distances(
        self,
        R: torch.Tensor[float],
        idxi: torch.Tensor[int],
        idxk: torch.Tensor[int],
        idxp: torch.Tensor[int],
    ) -> (torch.Tensor[float], torch.Tensor[int], torch.Tensor[int]):

        # Gather positions
        Ri = torch.gather(R, 0, idxi.view(-1, 1).repeat(1, 3))
        Rk = torch.gather(R, 0, idxk.view(-1, 1).repeat(1, 3))

        sum_distance = torch.sum((Ri - Rk)**2, dim=1)
        idxr = torch.squeeze(
            torch.where(
                sum_distance < self.max_rcut2,
                sum_distance,
                torch.zeros_like(sum_distance)
                )
            )

        # Interacting atom pair distances within cutoff
        Dik = torch.sqrt(torch.gather(sum_distance, 0, idxr))

        # Reduce the indexes to consider only interacting pairs
        idxi_r = torch.gather(idxi, 0, idxr)
        idxp_r = torch.gather(idxp, 0, idxr)

        return Dik, idxi_r, idxp_r

    def electrostatic_energy_per_atom_to_point_charge(
        self,
        Dik: torch.Tensor[float],
        Qai: torch.Tensor[float],
        Qak: torch.Tensor[float],
    ) -> torch.Tensor[float]:
        """
        Calculate electrostatic interaction between ML atom charge and MM point
        charge based on shifted Coulomb potential scheme
        """

        # Cutoff weighted reciprocal distance
        switchoff = self.switch_fn(Dik)

        # Shifted Coulomb energy
        qq = 2.0*self.kehalf*Qai*Qak
        Eele = qq/Dik - qq/self.mlmm_rcut*(2.0-Dik/self.mlmm_rcut)

        return torch.sum(switchoff*Eele)

    def run(
        self,
        mlmm_R: torch.Tensor[float],
        ml_Qa: torch.Tensor[float],
        mlmm_idxi: torch.Tensor[int],
        mlmm_idxk: torch.Tensor[int],
        mlmm_idxk_p: torch.Tensor[int],
    ) -> torch.Tensor[float]:
        """
        Calculates the electrostatic interaction between ML atoms in the center
        cell with all MM atoms in the center or imaginary non-bonded lists
        """

        # Calculate ML-MM atom distances
        mlmm_Dik, mlmm_idxi_r, mlmm_idxp_r = (
            self.calculate_mlmm_interatomic_distances(
                mlmm_R, mlmm_idxi, mlmm_idxk, mlmm_idxk_p)
            )
        mlmm_idxi_z = torch.gather(self.ml_idxp, 0, mlmm_idxi_r)

        # Get ML and MM charges
        ml_Qai_r = torch.gather(ml_Qa, 0, mlmm_idxi_z)
        ml_Qak_r = torch.gather(self.mlmm_atoms_charge, 0, mlmm_idxp_r)

        # Calculate electrostatic energy
        Eele = self.electrostatic_energy_per_atom_to_point_charge(
            mlmm_Dik, ml_Qai_r, ml_Qak_r)

        return Eele
