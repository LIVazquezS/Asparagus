import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.calculator import Calculator, CalculatorError
from ase.neighborlist import neighbor_list

import torch

from .. import model
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['ASE_Calculator']

class ASE_Calculator(Calculator):
    """
    ASE calculator interface for a Asparagus model potential.
    """
    
    def __init__(
        self,
        model_calculator: Union[object, List[object]],
        atoms: Optional[Union[object, List[object]]] = None,
        atoms_charge: Optional[Union[float, List[float]]] = None,
        implemented_properties: Optional[List[str]] = None,
        use_neighbor_list: Optional[bool] = None,
        label: Optional[str] = 'asparagus',
        **kwargs
    ):
        """
        Initialize ASE Calculator class.

        Parameters
        ----------

        model_calculator: (callable object, list of callable objects)
            NNP model calculator(s) to predict model properties. If an ensemble
            is given in form of a list of model calculators, the average value
            is returned as model prediction.
        atoms: (object, list(object)), optional, default None
            ASE Atoms object or list of ASE Atoms objects to which the 
            calculator will be attached.
        atoms_charge: (float, list(float)), optional, default None
            Total charge of the respective ASE Atoms object or objects.
            If the atoms charge is given as a float, the charge is assumed for
            all ASE atoms objects given.
        implemented_properties: (str, list(str)), optional, default None
            Properties predicted by the model calculator. If None, than
            all model properties (of the first model if ensemble) are 
            available.
        use_neighbor_list: bool, optional, default True
            If True, use the ASE neighbor list function to compute atom pair 
            indices within the model interaction cutoff regarding periodic 
            boundary conditions.
            If False, all possible atom pair indices are considered.
            If ASE Atoms object is periodic, neighbor list function will be
            used anyways.

        Returns
        -------
        callable object
            ASE calculator object
        """
        
        # Initialize parent Calculator class
        Calculator.__init__(self, label, **kwargs)
        
        ###################################
        # # # Check NNP Calculator(s) # # #
        ###################################

        # Assign NNP calculator model(s)
        if utils.is_array_like(model_calculator):
            self.model_calculator = None
            self.model_calculator_list = model_calculator
            self.model_ensemble = True
        else:
            self.model_calculator = model_calculator
            self.model_calculator_list = None
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
                            f"Model calculator {imodel:d} does not predict "
                            + f"property {prop:s}!\n" 
                            + "Specify 'implemented_properties' with "
                            + "properties all model calculator support.")

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

        #################################
        # # # Check ASE Atoms Input # # #
        #################################

        # Check atoms input
        self.atoms = None
        self.atoms_list = None
        self.set_atoms(atoms, atoms_charge=atoms_charge)
        
        # Assign calculator
        if self.atoms_ensemble and self.atoms_list is not None:
            for atoms_i in self.atoms_list:
                atoms_i.calc = self
        elif self.atoms is not None:
            self.atoms.calc = self
        
        # Flag for the use atoms neighbor list function to potentially reduce 
        # the number atom pairs
        if use_neighbor_list is None:
            self.use_neighbor_list = True
        else:
            if utils.is_bool(use_neighbor_list):
                self.use_neighbor_list = use_neighbor_list
            else:
                raise SyntaxError(
                    "Input flag 'use_neighbor_list' must be a boolean!")
        
        # Prepare model calculator input
        self.atoms_batch = self.initialize_model_input()


    def set_atoms(self, atoms, atoms_charge=None):
        """
        Assign atoms and calculator
        """
        print('set atoms top', atoms)
        if atoms is None:
            self.atoms = None
            self.atoms_list = None
            self.atoms_ensemble = False
        elif utils.is_array_like(atoms):
            # If Atoms already assigned check compatibility
            print('check list', self.atoms, self.atoms_list)
            if self.atoms is not None:
                raise CalculatorError(
                    "New Atoms object is not compatible with assigned "
                    + "Atoms object!")
            elif self.atoms_list is not None:
                equal = np.all(
                    [
                        np.logical_and(
                            np.all(atoms.numbers == self.atoms.numbers),
                            np.all(atoms.pbc == self.atoms.pbc))
                        for atom in self.atoms_list
                    ])
                if not equal:
                    raise CalculatorError(
                        "New Atoms object is not compatible with assigned "
                        + "Atoms object!")
            # Assign Atoms object ensemble to calculator
            self.atoms = None
            self.atoms_list = atoms
            self.atoms_ensemble = True
        else:
            # If Atoms already assigned check compatibility
            if self.atoms is not None:
                if self.atoms_ensemble:
                    equal = False
                else:
                    equal = np.logical_and(
                        np.all(atoms.numbers == self.atoms.numbers),
                        np.all(atoms.pbc == self.atoms.pbc))
                if not equal:
                    raise CalculatorError(
                        "New Atoms object is not compatible with assigned "
                        + "Atoms object!")
            # Assign Atoms object to calculator
            self.atoms = atoms
            self.atoms_list = None    
            self.atoms_ensemble = False
            
        # Check atoms charge input
        if self.atoms is None and self.atoms_list is None:
            self.atoms_charge = None
        elif atoms_charge is None:
            if self.atoms_ensemble:
                self.atoms_charge = [0.0]*len(self.atoms_list)
            else:
                self.atoms_charge = 0.0
        elif utils.is_array_like(atoms_charge):
            if self.atoms_ensemble:
                if len(atoms_charge) != len(self.atoms_list):
                    raise SyntaxError(
                        f"Number of provided atoms charges "
                        + f"{len(atoms_charge):d} is different than the "
                        + f"number of given atoms systems {len(self.atoms):d}!"
                        )
                else:
                    self.atoms_charge = atoms_charge
            else:
                raise SyntaxError(
                    "Just provide one float number for the charge of just "
                    + "one ASE Atoms object!\n"
                    + "A list for atoms charges were given."
                    )
        else:
            if self.atoms_ensemble:
                self.atoms_charge = [atoms_charge]*len(self.atoms_list)
            else:
                self.atoms_charge = atoms_charge


    def initialize_model_input(self):
        """
        Initial preparation of the model calculator input
        """
        
        # Initialize model calculator input
        atoms_batch = {}
        
        # Check if atoms object(s) are given
        if self.atoms is None and self.atoms_list is None:
            return atoms_batch
        
        # Case: ASE atoms object ensemble
        if self.atoms_ensemble:
            
            # Number of atoms system and atoms number per system
            atoms_batch['Nsys'] = len(self.atoms_list)
            Natoms = torch.tensor(
                [len(atoms) for atoms in self.atoms_list], dtype=torch.int64)
            Natoms_sum = torch.sum(Natoms)
            atoms_batch['Natoms_cumsum'] = torch.cat(
                [
                    torch.zeros((1,), dtype=torch.int64), 
                    torch.cumsum(Natoms, dim=0)
                ],
                dim=0)
            
            # Number of atoms
            atoms_batch['atoms_number'] = Natoms

            # Atom positions
            atoms_batch['positions'] = torch.zeros(
                [Natoms_sum, 3], dtype=torch.float64)
            
            # Atomic number
            atomic_numbers = torch.empty(Natoms_sum, dtype=torch.int64)
            i_atom = 0
            for atoms in self.atoms_list:
                for z_atom in atoms.get_atomic_numbers():
                    atomic_numbers[i_atom] = z_atom
                    i_atom += 1
            atoms_batch['atomic_numbers'] = atomic_numbers

            # Atom segment indices
            atoms_batch['atoms_seg'] = torch.repeat_interleave(
                torch.arange(
                    atoms_batch['Nsys'], dtype=torch.int64), 
                repeats=Natoms, dim=0)
            
            # Total atomic system charge
            atoms_batch['charge'] = torch.tensor(
                self.atoms_charge, dtype=torch.float64)

            # Changing model calculator input for atoms object
            atoms_batch = self.update_model_input(atoms_batch)
            
            
        # Case: Single ASE atoms object
        else:
            
            # Number of atoms
            Natoms = len(self.atoms)
            atoms_batch['atoms_number'] = torch.tensor(
                [Natoms], dtype=torch.int64)

            # Atom positions
            atoms_batch['positions'] = torch.zeros(
                [Natoms, 3], dtype=torch.float64)

            # Atomic number
            atoms_batch['atomic_numbers'] = torch.tensor(
                self.atoms.get_atomic_numbers(), dtype=torch.int64)

            # Atom segment indices, just one atom segment allowed
            atoms_batch['atoms_seg'] = torch.zeros(
                Natoms, dtype=torch.int64)

            # Total atomic system charge
            atoms_batch['charge'] = torch.tensor(
                [self.atoms_charge], dtype=torch.float64)
            
            # Changing model calculator input for atoms object
            atoms_batch = self.update_model_input(atoms_batch)
        
        

        return atoms_batch


    def update_model_input(self, atoms_batch):
        """
        Update model calculator input.
        """
        
        if self.atoms_ensemble:

            # Update atom positions
            i_atom = 0
            for atoms in self.atoms_list:
                for p_atom in torch.tensor(
                        atoms.get_positions(), dtype=torch.float64
                ):
                    atoms_batch['positions'][i_atom, :] = p_atom
                    i_atom += 1
            
            # Create and assign atom pair indices and periodic offsets
            idx_i, idx_j, pbc_offset = [], [], []
            for shift, (i_atom, atoms) in zip(
                    atoms_batch['Natoms_cumsum'][:-1], 
                    enumerate(self.atoms_list)
                ):
                if self.use_neighbor_list or any(atoms.get_pbc()):
                    atoms_idx_i, atoms_idx_j, atoms_pbc_offset = neighbor_list(
                        'ijS',
                        atoms,
                        self.interaction_cutoff,
                        self_interaction=False)
                    atoms_idx_i = torch.tensor(atoms_idx_i, dtype=torch.int64)
                    atoms_idx_j = torch.tensor(atoms_idx_j, dtype=torch.int64)
                    atoms_pbc_offset = torch.tensor(
                        atoms_pbc_offset, dtype=torch.float64)
                else:
                    idx = torch.arange(
                        end=atoms_batch['atoms_number'][i_atom], 
                        dtype=torch.int64)
                    atoms_idx_i = idx.repeat(
                        atoms_batch['atoms_number'][i_atom] - 1)
                    atoms_idx_j = torch.roll(idx, -1, dims=0)
                    atoms_pbc_offset = torch.zeros(
                        (atoms_batch['atoms_number'][i_atom], 3),
                        dtype=torch.float64)
                idx_i.append(atoms_idx_i + shift)
                idx_j.append(atoms_idx_j + shift)
                pbc_offset.append(atoms_pbc_offset)
            atoms_batch['idx_i'] = torch.cat(idx_i, dim=0)
            atoms_batch['idx_j'] = torch.cat(idx_j, dim=0)
            atoms_batch['pbc_offset'] = torch.cat(pbc_offset, dim=0)

            # Atom pairs segment
            Npairs = torch.tensor(
                [len(atoms_idx_i) for atoms_idx_i in idx_i])
            atoms_batch['pairs_seg'] = torch.repeat_interleave(
                torch.arange(atoms_batch['Nsys'], dtype=torch.int64), 
                repeats=Npairs, dim=0)

        else:

            # Atom positions
            atoms_batch['positions'] = torch.tensor(
                self.atoms.get_positions(), dtype=torch.float64)

            # Create and assign atom pair indices and periodic offsets
            if self.use_neighbor_list or any(self.atoms.get_pbc()):
                idx_i, idx_j, pbc_offset = neighbor_list(
                    'ijS',
                    self.atoms,
                    self.interaction_cutoff,
                    self_interaction=False)
                atoms_batch['idx_i'] = torch.tensor(idx_i, dtype=torch.int64)
                atoms_batch['idx_j'] = torch.tensor(idx_j, dtype=torch.int64)
                atoms_batch['pbc_offset'] = torch.tensor(
                    pbc_offset, dtype=torch.float64)
            else:
                idx = torch.arange(
                    end=atoms_batch['atoms_number'][0], dtype=torch.int64)
                atoms_batch['idx_i'] = idx.repeat(
                    atoms_batch['atoms_number'][0] - 1)
                atoms_batch['idx_j'] = torch.roll(idx, -1, dims=0)
                atoms_batch['pbc_offset'] = torch.zeros(
                    (atoms_batch['atoms_number'], 3), dtype=torch.float64)
            
            # Atom pairs segment index, also just one atom pair segment allowed
            atoms_batch['pairs_seg'] = torch.zeros_like(
                atoms_batch['idx_i'], dtype=torch.int64)
        
        return atoms_batch


    def calculate(
        self, 
        atoms: Optional[Union[object, List[object]]] = None,
        atoms_charge: Optional[Union[float, List[float]]] = None,
        properties: List[str] = None,
        **kwargs
    ):
        """
        Calculate model properties
        
        Parameters
        ----------
        atoms: (object, list(object)), optional, default None
            Optional ASE Atoms object or list of ASE Atoms objects to which the
            properties will be calculated. If given, atoms setup to prepare 
            model calculator input will be run again.
        atoms_charge: (float, list(float)), optional, default None
            Optional total charge of the respective ASE Atoms object or 
            objects. If the atoms charge is given as a float, the charge is 
            assumed for all ASE atoms objects given.
        properties: list(str), optional, default None
            List of properties to be calculated. If None, all implemented
            properties will be calculated (will be anyways ...).
        """
        
        # Check atoms input
        if atoms is None and self.atoms is None and self.atoms_list is None:
            raise CalculatorSetupError(
                f"ASE atoms object is not defined!")
        elif atoms is None:
            self.atoms_batch = self.update_model_input(self.atoms_batch)
        else:
            self.set_atoms(atoms, atoms_charge=atoms_charge)
            self.atoms_batch = self.initialize_model_input()

        if self.model_ensemble:
            for calc in self.model_calculator_list:
                prediction = calc(self.atoms_batch)
        else:
            prediction = self.model_calculator(self.atoms_batch)
        print(prediction)

