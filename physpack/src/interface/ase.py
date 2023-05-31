import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase

from .. import utils
from .. import debug

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_ase_calculator']


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
            calculator = ase_calculator_avaiable[calculator.lower()](
                **calculator_args)
        except TypeError as error:
            logger.error(error)
            raise TypeError(
                f"ASE calculator '{calculator}' does not accept " +
                f"arguments in 'calculator_args'")

    # Retrun ASE calculator
    return calculator

