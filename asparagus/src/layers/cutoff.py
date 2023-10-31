#TODO: Add more options of cutoff functions. Current Cutoff only does regular
# physnet cutoff
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

__all__ = ['get_cutoff_fn', 'Poly6_cutoff', 'cutoff_CHARMM']

#======================================
# Cutoff functions
#======================================

class Poly6_cutoff(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order.

    f(x) = 1 - 6*x**5 + 15*x**4 - 10*x**3 with x = distance/cutoff
    """

    def __init__(
        self,
        cutoff: float,
        dtype: Optional[object] = torch.float64,
    ):
        """
        Parameters
        ----------
        cutoff: float
            Cutoff distance

        Returns
        -------
        float
            Cutoff value [0,1]
        """

        super().__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            torch.zeros_like(x))

class cutoff_CHARMM(torch.nn.Module):
    ''' Switch function for electrostatic interaction (switches between
        shielded and unshielded electrostatic interaction) '''

    def __init__(
        self,
        mlmm_rcut: float,
        mlmm_width: float,
        dtype: Optional[object] = torch.float64,
    ):

        super().__init__()

        self.mlmm_rcut = mlmm_rcut
        self.mlmm_width = mlmm_width
        # Set cutoff value in the register for model parameters
        self.register_buffer("mlmm_rcut", torch.tensor([mlmm_rcut], dtype=dtype))
        self.register_buffer("mlmm_width", torch.tensor([mlmm_width], dtype=dtype))

    def forward(
        self,
        Dmlmm: torch.Tensor
    ) -> torch.Tensor:

        x = (Dmlmm - self.mlmm_rcut + self.mlmm_width)/self.mlmm_width

        cutoff = torch.where(Dmlmm<self.mlmm_rcut,torch.ones_like(Dmlmm), torch.zeros_like(Dmlmm))

        cutoff = torch.where(
            torch.logical_and((Dmlmm>self.mlmm_rcut-self.mlmm_width),(Dmlmm<self.mlmm_rcut)),
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,cutoff)

        return cutoff

#======================================
# Function assignment
#======================================


functions_avaiable = {
    'default'.lower(): Poly6_cutoff,
    'Poly6'.lower(): Poly6_cutoff,
    'PhysNet_CutOff'.lower(): Poly6_cutoff,
    'CHARMM_CutOff'.lower(): cutoff_CHARMM,
    'CHARMM'.lower(): cutoff_CHARMM
    }


def get_cutoff_fn(
    name: Optional[Union[object, str]] = None
):
    """
    Get cutoff function by defined name.

    Parameters
    ----------

        name: (str, object), optional, default None
            If name is a str than it checks for the corresponding cutoff
            function and return the function object.
            The input will be given if it is already a function object.
            If None, then default cutoff function is used.

    Returns
    -------
        object
            Cutoff function object
    """

    # Check for default option
    if name is None:
        name = 'default'

    # Get cutoff function
    if utils.is_callable(name):

        return name

    if utils.is_string(name):

        if name.lower() in functions_avaiable.keys():
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Cutoff function input '{name}' is not valid!" +
                "Choose from:\n" +
                str(functions_avaiable.keys()))

    else:

        raise ValueError(
            f"Cutoff function input of type '{type(name)}' " +
            "is not valid! Input 'name' has to be an object or 'str' from;\n" +
            str(functions_avaiable.keys()))
