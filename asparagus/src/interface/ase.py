import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase

from .. import utils
from .. import debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'ase_calculator_units', 'get_ase_calculator', 'ase_calculate_properties']


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

def ase_calculate_properties(
    system, 
    properties,
    ase_properties,
):
    """
    ASE Calculator interface

    Parameters
    ----------
    system: ASE Atoms object
        ASE Atoms object with assigned ASE calculator
    properties: list(str)
        List of properties to compute
    ase_properties: list(str)
        List of properties to compute directly by ASE calculator

    Returns
    -------
    dict
        Computed properties
    """
    
    #system.get_properties()
    
    #system._calc.calculate_properties(
        #system, sample_properties)
    pass
