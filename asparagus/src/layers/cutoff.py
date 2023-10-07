#TODO: Add more options of cutoff functions. Current Cutoff only does regular
# physnet cutoff
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

__all__ = ['get_cutoff_fn', 'Poly6_cutoff']

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


#======================================
# Function assignment
#======================================


functions_avaiable = {
    'default'.lower(): Poly6_cutoff,
    'Poly6'.lower(): Poly6_cutoff,
    'PhysNet_CutOff'.lower(): Poly6_cutoff,
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
