import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase.calculators.calculator as ase_calc
from ase.neighborlist import neighbor_list

import torch

from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['ASE_Calculator']


class ASE_Calculator(ase_calc.Calculator):
    """
    ASE calculator interface for a Asparagus model potential.
    """

    default_parameters = {
        "method": "Asparagus",
    }

    def __init__(
        self,
        model_calculator: Union[object, List[object]],
        atoms: Optional[object] = None,
        atoms_charge: Optional[float] = None,
        implemented_properties: Optional[List[str]] = None,
        use_neighbor_list: Optional[bool] = None,
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
        atoms: ASE Atoms object, optional, default None
            ASE Atoms object to which the calculator will be attached.
        atoms_charge: float, optional, default 0.0
            Total charge of the respective ASE Atoms object.
        implemented_properties: (str, list(str)), optional, default None
            Properties predicted by the model calculator. If None, then
            all model properties (of the first model if ensemble) are
            available.
        use_neighbor_list: bool, optional, default True
            If True, use the ASE neighbor list function to compute atom pair
            indices within the model interaction cutoff regarding periodic
            boundary conditions.
            If False, all possible atom pair indices are considered.
            If ASE Atoms object is periodic, neighbor list function will be
            used anyway.

        Returns
        -------
        callable object
            ASE calculator object
        """

        # Initialize parent Calculator class
        ase_calc.Calculator.__init__(self, atoms=atoms, **kwargs)

        ###################################
        # # # Check NNP Calculator(s) # # #
        ###################################

        # Assign NNP calculator model(s)
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

        # Get property unit conversions from model units to ASE units
        self.model_unit_properties = (
            self.model_calculator.model_unit_properties)
        self.model2ase_unit_conversion = {}
        # Positions unit (None = ASE units by default)
        conversion, _ = utils.check_units(
            None, self.model_unit_properties['positions'])
        self.model2ase_unit_conversion['positions'] = conversion
        # Implemented property units (None = ASE units by default)
        for prop in self.implemented_properties:
            conversion, _ = utils.check_units(
                None, self.model_unit_properties[prop])
            self.model2ase_unit_conversion[prop] = conversion

        # Set atoms charge
        self.set_atoms_charge(atoms_charge, initialize=False)

        # Set ASE atoms object
        self.set_atoms(atoms, initialize=True)

        return

    def set_atoms(self, atoms, initialize=False):
        """
        Assign atoms object to calculator and prepare input.
        """
        if atoms is None:
            self.atoms = None
        else:
            self.atoms = atoms.copy()

        # Prepare model calculator input
        if initialize:
            self.atoms_batch = self.initialize_model_input()

    def set_atoms_charge(self, atoms_charge, initialize=False):
        """
        Assing atoms charge.
        """
        if atoms_charge is None:
            self.atoms_charge = 0.0
        elif utils.is_numeric(atoms_charge):
            self.atoms_charge = float(atoms_charge)
        else:
            raise SyntaxError(
                "Provide a float for the charge of the atoms system!")

        # Prepare model calculator input
        if initialize:
            self.atoms_batch = self.initialize_model_input()

    def initialize_model_input(self):
        """
        Initial preparation for an ASE Atoms objects
        """

        # Initialize model calculator input
        atoms_batch = {}

        # Check if atoms object(s) are given
        if self.atoms is None:
            return atoms_batch

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
        Update model calculator input for an ASE Atoms object.
        """

        # If model input is not initialized
        if not atoms_batch:
            atoms_batch = self.initialize_model_input()
            return atoms_batch

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

    def initialize_model_input_list(self, atoms_list, atoms_charge):
        """
        Initial preparation for a list of ASE Atoms objects.
        """

        # Initialize model calculator input
        atoms_batch = {}

        # Check if atoms object(s) are given
        if atoms_list is None:
            return atoms_batch

        # Number of atoms system and atoms number per system
        atoms_batch['Nsys'] = len(atoms_list)
        Natoms = torch.tensor(
            [len(atoms) for atoms in atoms_list], dtype=torch.int64)
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
        for atoms in atoms_list:
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
            atoms_charge, dtype=torch.float64)

        # Changing model calculator input for atoms object
        atoms_batch = self.update_model_input_list(atoms_batch, atoms_list)

        return atoms_batch

    def update_model_input_list(self, atoms_batch, atoms_list):
        """
        Update model calculator input for a list of ASE Atoms objects.
        """

        # Update atom positions
        i_atom = 0
        for atoms in atoms_list:
            for p_atom in torch.tensor(
                    atoms.get_positions(), dtype=torch.float64):
                atoms_batch['positions'][i_atom, :] = p_atom
                i_atom += 1

        # Create and assign atom pair indices and periodic offsets
        idx_i, idx_j, pbc_offset = [], [], []
        for shift, (i_atom, atoms) in zip(
                atoms_batch['Natoms_cumsum'][:-1],
                enumerate(atoms_list)):
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

        return atoms_batch


    def calculate(
        self,
        atoms: Optional[Union[object, List[object]]] = None,
        atoms_charge: Optional[Union[float, List[float]]] = None,
        properties: List[str] = None,
        system_changes: List[str] = ase_calc.all_changes,
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
        atoms_list = False
        if atoms is None and self.atoms is None:

            raise ase_calc.CalculatorSetupError(
                "ASE atoms object is not defined!")

        elif atoms is None:

            self.atoms_batch = self.update_model_input(self.atoms_batch)
            atoms_batch = self.atoms_batch

        else:

            # If prediction for an atoms ensemble is requested
            if utils.is_array_like(atoms):

                atoms_list = True

                # Check atoms charges
                if atoms_charge is None:
                    atoms_charge = [0.0]*len(atoms)
                elif utils.is_numeric(atoms_charge):
                    atoms_charge = [atoms_charge]*len(atoms)
                elif utils.is_numeric_array(atoms_charge):
                    if len(atoms_charge) != len(atoms):
                        raise SyntaxError(
                            "Number of provided atoms charges "
                            + f"{len(atoms_charge):d} does not match the "
                            + "number of given atoms systems "
                            + f"{len(self.atoms):d}!")
                else:
                    raise SyntaxError(
                        "Provide one float or a list of float for the charges "
                        + "of the atoms system!\n")
                # Get model input for the atoms list
                atoms_batch = self.initialize_model_input_list(
                    atoms, atoms_charge)

            # For a single atoms object, reinitialze
            else:

                if properties is None:
                    properties = self.implemented_properties
                ase_calc.Calculator.calculate(
                    self, atoms, properties, system_changes)
                if utils.is_numeric(atoms_charge):
                    self.atoms_charge = float(atoms_charge)
                elif self.atoms_charge is None:
                    self.atoms_charge = 0.0
                self.atoms_batch = self.initialize_model_input()
                atoms_batch = self.atoms_batch

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

        # Convert model properties
        self.results = {}
        if atoms_list:
            # TODO Resolve results per atoms object
            for prop in self.implemented_properties:
                all_results = (
                    results[prop].detach().numpy()
                    *self.model2ase_unit_conversion[prop])
                self.results[prop] = self.resolve_atoms(
                    all_results, atoms_batch, prop)
                
        else:
            for prop in self.implemented_properties:
                self.results[prop] = (
                    results[prop].detach().numpy()
                    *self.model2ase_unit_conversion[prop])

        return self.results

    def resolve_atoms(
        self,
        all_results: List[float],
        atoms_batch: Dict[str, Any],
        prop: str,
    ):

        # Compare result shape with number of atom systems, atom numbers and
        # pair numbers
        if all_results.shape[0] == atoms_batch['atoms_number'].shape[0]:
            return [
                all_results[isys] 
                for isys, _ in enumerate(atoms_batch['atoms_number'])]
        elif all_results.shape[0] == atoms_batch['atomic_numbers'].shape[0]:
            atoms_results = []
            iatoms = 0
            for Natoms in atoms_batch['atoms_number']:
                atoms_results.append(
                    all_results[iatoms:(iatoms + Natoms)])
                iatoms = iatoms + Natoms
            return atoms_results
        elif all_results.shape[0] == atoms_batch['pairs_seg'].shape[0]:
            atoms_results = [
                [] for _ in atoms_batch['atoms_number']]
            for ipair, isys in enumerate(atoms_batch['pairs_seg']):
                atoms_results[isys].append(all_results[ipair])
            return atoms_results
        else:
            raise SyntaxError(
                f"Model result of property '{prop:s}' could not be resolved!")
        
        
        
