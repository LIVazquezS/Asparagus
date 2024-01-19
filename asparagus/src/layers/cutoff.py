# TODO: Add more options of cutoff functions. Current Cutoff only does regular
# PhysNet cutoff
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

__all__ = ['get_cutoff_fn', 'Cutoff_poly6', 'Cutoff_poly6_width']

#======================================
# Cutoff functions
#======================================


class Cutoff_poly6(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order.

    $$f(x) = 1 - 6x^{5} + 15x^{4} - 10x^{3}$$
    with $$x = distance/cutoff$$

    Parameters
    ----------
    cutoff: float
        Cutoff distance

    Returns
    -------
    float
        Cutoff value [0,1]


    """

    def __init__(
        self,
        cutoff: float,
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super().__init__()
        
        #self.cutoff = cutoff
        
        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))
        
    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        '''

        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Distance tensor of shape (N, M) where N is the number of atoms and
            M is the number of neighbors.

        Returns
        -------

        '''

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            torch.zeros_like(x))


class Cutoff_poly6_width(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order,
    within the range cutoff - width < x < cutoff

    $$f(x) = 1 - 6x^{5} + 15x^{4} - 10x^{3}$$
    with $$(x - cutoff - width) = distance/width$$

    **Note**: This function is for CHARMM potential.

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    width: float
        Cutoff width defining cutoff range (cutoff - width) to cutoff

    Returns
    -------
    float
        Cutoff value [0,1]


    """

    def __init__(
        self,
        cutoff: float,
        width: float,
        dtype: Optional[object] = torch.float64,
    ):


        super().__init__()

        #self.cutoff = cutoff
        #self.width = width

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))
        self.register_buffer("width", torch.tensor([width], dtype=dtype))

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        '''

        Forward pass of the cutoff function.
        Parameters
        ----------
        distance : torch.Tensor


        Returns
        -------

        '''

        x = (distance - self.cutoff + self.width)/self.width

        poly6_width = torch.where(
            distance < self.cutoff,
            torch.ones_like(distance),
            torch.zeros_like(distance)
            )

        poly6_width = torch.where(
            torch.logical_and(
                (distance > self.cutoff - self.width),
                (distance < self.cutoff)
            ),
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            poly6_width)

        return poly6_width

#======================================
# Function assignment
#======================================

functions_avaiable = {
    'default'.lower(): Cutoff_poly6,
    'Poly6'.lower(): Cutoff_poly6,
    'PhysNet'.lower(): Cutoff_poly6,
    'PhysNet_cutoff'.lower(): Cutoff_poly6,
    'Poly6_width'.lower(): Cutoff_poly6_width,
    'CHARMM'.lower(): Cutoff_poly6_width,
    'CHARMM_cutoff'.lower(): Cutoff_poly6_width,
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
