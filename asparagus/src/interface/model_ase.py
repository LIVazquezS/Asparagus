import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.calculator import Calculator
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
        model_calculators: Union[object, List[object]],
        implemented_properties: List[str] = None,
        label: Optional[str] = 'asparagus',
        atoms: Optional[object] = None,
        **kwargs
    ):
        """
        Initialize ASE Calculator class.

        Parameters
        ----------

        model_calculators: (callable object, list of callable objects)
            NNP model calculator(s) to predict model properties. If an ensemble
            is given in form of a list of model calculators, the average value
            is returned as model prediction.
        implemented_properties: list(str), optional, default None
            Properties predicted by the model calculator. If None, than
            all model properties (of the first model if ensemble) are 
            available.
        atoms: ASE Atoms object, optional, default None
            Optional Atoms object to which the calculator will be attached.

        Returns
        -------
        callable object
            ASE calculator object
        """
        
        # Initialize parent Calculator class
        Calculator.__init__(self, label, atoms, **kwargs)
        
        ###################################
        # # # Check NNP Calculator(s) # # #
        ###################################

        # Assign NNP calculator model(s)
        if utils.is_array_like(model_calculators):
            self.model_calculators = model_calculators
            self.model_ensemble = True
        else:
            self.model_calculators = [model_calculators]
            self.model_ensemble = False

        # Set implemented properties
        if implemented_properties is None:
            self.implemented_properties = (
                self.model_calculators[0].model_properties)
        else:
            self.implemented_properties = implemented_properties

        # Check model properties and set evaluation mode
        for imodel, model_calculator in enumerate(self.model_calculators):
            for prop in self.implemented_properties:
                if prop not in model_calculator.model_properties:
                    raise SyntaxError(
                        f"Model calculator {imodel:d} does not predict "
                        + f"property {prop:s}!\n" 
                        + "Specify 'implemented_properties' with properties "
                        + "all model calculator support.")

        # Get model interaction cutoff and set evaluation mode
        self.interaction_cutoff = 0.0
        for imodel, model_calculator in enumerate(self.model_calculators):
            cutoff = model_calculator.model_interaction_cutoff
            if self.interaction_cutoff < cutoff:
                self.interaction_cutoff = cutoff
            model_calculator.eval()

        # If atoms object given, prepare model calculator input
        if self.atoms is not None:
            self.initial_model_input()


    def initialize_model_input(self, atoms_charge=0.0):
        """
        Initial preparation of the model calculator input
        """
        
        # Initialize model calculator input
        self.atoms_batch = {}
        
        # Constant model calculator input for atoms object
        
        # Atoms number
        Natoms = len(self.atoms)
        self.atoms_batch['atoms_number'] = torch.tensor(
            [Natoms], dtype=torch.int64)
        
        # Initialize atom positions
        self.atoms_batch['positions'] = torch.zeros(
            [Natoms, 3], dtype=torch.float64)
        
        # Atomic number
        self.atoms_batch['atomic_numbers'] = torch.tensor(
            self.atoms.get_atomic_numbers(), dtype=torch.int64)

        # Atom segment indices, just one atom segment allowed
        self.atoms_batch['atoms_seg'] = torch.zeros(Natoms, dtype=torch.int64)

        # Total atomic system charge
        self.atoms_batch['charge'] = torch.tensor(
            [atoms_charge], dtype=torch.float64)

        # Changing model calculator input for atoms object
        self.update_model_input()


    def update_model_input(self):
        """
        Update model calculator input.
        """
        
        # Update atom positions
        self.atoms_batch['positions'] = torch.tensor(
            self.atoms.get_positions(), dtype=torch.float64)

        # Create and assign atom pair indices and periodic offsets
        idx_i, idx_j, pbc_offset = neighbor_list(
            'ijS',
            self.atoms,
            self.interaction_cutoff,
            self_interaction=False)
        self.atoms_batch['idx_i'] = torch.tensor(idx_i, dtype=torch.int64)
        self.atoms_batch['idx_j'] = torch.tensor(idx_j, dtype=torch.int64)
        self.atoms_batch['pbc_offset'] = torch.tensor(
            pbc_offset, dtype=torch.float64)

        # Atom pairs segment index, also just one atom pair segment allowed
        self.atoms_batch['pairs_seg'] = torch.zeros(
            len(idx_i), dtype=torch.int64)


    def calculate(
        self, 
        atoms: Optional[object] = None,
        properties: List[str] = None,
        **kwargs
    ):
        """
        Calculate model properties
        
        Parameters
        ----------
        atoms: ASE Atoms object, optional, default None
            Optional Atoms object of which the properties will be calculated.
            If given, atoms setup to prepare model calculator input will be 
            run again.
        properties: list(str), optional, default None
            List of properties to be calculated. If None, all implemented
            properties will be calculated (will be anyways ...).
        """

        # Check atoms input
        if atoms is not None:
            # Reinitialize model calculator input
            self.atoms = atoms.copy()
            self.initialize_model_input()
        elif self.atoms is None:
            raise CalculatorSetupError(
                f"ASE atoms object is not defined!")
        else:
            # Update model calculator input
            self.update_model_input()

        for imodel, model_calculator in enumerate(self.model_calculators):
            prediction = model_calculator(self.atoms_batch)
            print(prediction)
        
