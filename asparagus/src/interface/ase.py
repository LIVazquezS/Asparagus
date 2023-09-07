import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.orca import ORCA

from .. import utils
from .. import debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'ase_calculator_units', 'get_ase_calculator', 'get_ase_properties']


# ======================================
# ASE Calculator Units
# ======================================

ase_calculator_units = {
    'positions':        'Ang',
    'energy':           'eV',
    'forces':           'eV/Ang',
    'hessian':          'eV/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'eAng',
    }


# ======================================
# Calculator Assignment
# ======================================

ase_calculator_avaiable = {
    'XTB'.lower(): debug.XTB,
    'ORCA'.lower(): ORCA,
    }

def get_ase_calculator(
    calculator,
    calculator_args
):
    """
    ASE Calculator interface

    Parameters
    ----------
    calculator: (str, object)
        Calculator label of an ASE calculator to initialize or an
        ASE calculator object directly returned
    calculator_args: dict
        ASE calculator arguments if ASE calculator will be initialized

    Returns
    -------
    callable object
        ASE Calculator object to compute atomic systems
    """

    # Initialize calculator name parameter
    calculator_tag = None

    # In case of calculator label, initialize ASE calculator
    if utils.is_string(calculator):

        # Check avaiability
        if calculator.lower() not in ase_calculator_avaiable:
            raise ValueError(
                f"ASE calculator '{calculator}' is not avaiable!" +
                f"Choose from:\n" +
                str(ase_calculator_avaiable.keys()))

        # initialize ASE calculator
        try:
            calculator_tag = calculator
            calculator = ase_calculator_avaiable[calculator.lower()](
                **calculator_args)
        except TypeError as error:
            logger.error(error)
            raise TypeError(
                f"ASE calculator '{calculator}' does not accept " +
                f"arguments in 'calculator_args'")

    else:
        
        # Check for calculator name parameter in calculator class
        if hasattr(calculator, 'calculator_tag'):
            calculator_tag = calculator.calculator_tag
        
    # Retrun ASE calculator and name label
    return calculator, calculator_tag


# ======================================
# ASE Calculator Properties
# ======================================

def get_ase_properties(
    system, 
    calc_properties,
):
    """
    ASE Calculator interface

    Parameters
    ----------
    system: ASE Atoms object
        ASE Atoms object with assigned ASE calculator
    calc_properties: list(str)
        List of computed properties to return

    Returns
    -------
    dict
        Property dictionary containing:
        atoms number, atomic numbers, positions, cell size, periodic boundary 
        conditions, total charge and computed properties.
    """
    
    # Initialize property dictionary
    properties = {}
    
    # Atoms number
    properties['atoms_number'] = system.get_global_number_of_atoms()
    
    # Atomic numbers
    properties['atomic_numbers'] = system.get_atomic_numbers()
    
    # Atomic numbers
    properties['positions'] = system.get_positions()
    
    # Periodic boundary conditions
    properties['cell'] = list(system.get_cell())[0]
    properties['pbc'] = system.get_pbc()
    
    # Total charge
    # Try from calculator paramters
    if 'charge' in system._calc.parameters:
        charge = system._calc.parameters['charge']
    else:
        charge = None
    # If charge is still None, try from computed atomic charges
    if charge is None:
        try:
            atomic_charges = system.get_charges()
            charge = round(np.sum(system.get_charges()))
        except ase.calculators.calculator.PropertyNotImplementedError:
            charge = None
    # If charge is still None, assume neutral charge
    if charge is None:
        charge = 0
    properties['charge'] = charge
    
    for ip, prop in enumerate(calc_properties):
        if prop in properties:
            continue
        properties[prop] = system._calc.results.get(prop)
    
    return properties
