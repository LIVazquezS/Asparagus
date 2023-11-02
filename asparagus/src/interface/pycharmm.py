#Basic imports
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

#ASE imports
import ase.units as units

#NN imports
from .. import utils

#PyCHARMM imports
import pycharmm

#Start code

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# functions to be implemented
# Get the calculator
# Units conversion between asparagus and pycharmm
# Get properties.


# ======================================
# Charmm Calculator Units
# ======================================

CHARMM_calculator_units = {
    'positions':        'Ang',
    'energy':           'kcal/mol',
    'forces':           'kcal/mol/Ang',
    'hessian':          'kcal/mol/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'eAng',
    }


