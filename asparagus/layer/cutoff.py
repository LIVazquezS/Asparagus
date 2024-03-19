
import math
from typing import Optional, Union, Callable

import torch

from .. import utils

__all__ = [
    'get_cutoff_fn', 'Poly6Cutoff', 'Poly6Cutoff_Width', 'CosineCutoff',
    'CosineCutoff_Width']

#======================================
# Cutoff functions
#======================================


class Poly6Cutoff(torch.nn.Module):
    """
    2nd derivative smooth polynomial cutoff function of 6th order.

    $$f(x) = 1 - 6x^{5} + 15x^{4} - 10x^{3}$$
    with $$x = distance/cutoff$$

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(Poly6Cutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            torch.zeros_like(distance))


class Poly6Cutoff_Width(torch.nn.Module):
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
        Cutoff width defining cutoff range (cutoff - width) to (cutoff)
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        width: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(Poly6Cutoff_Width, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))
        self.register_buffer("width", torch.tensor([width], dtype=dtype))

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """

        x = (distance - self.cutoff + self.width)/self.width

        cutoff = torch.where(
            distance < self.cutoff,
            torch.ones_like(distance),
            torch.zeros_like(distance)
            )

        cutoff = torch.where(
            torch.logical_and(
                (distance > self.cutoff - self.width),
                (distance < self.cutoff)
            ),
            1.0 + ((-6.0*x + 15.0)*x - 10.0)*x**3,
            cutoff)

        return cutoff 


class CosineCutoff(torch.nn.Module):
    """
    Behler-style cosine cutoff module, 
    within the range cutoff - width < x < cutoff
    [source https://github.com/atomistic-machine-learning/schnetpack/blob/
        master/src/schnetpack/nn/cutoff.py]

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(CosineCutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

        return

    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """

        x = distance/self.cutoff

        return torch.where(
            x < 1.0,
            0.5*(torch.cos(x*math.pi) + 1.0),
            torch.zeros_like(distance))


class CosineCutoff_Width(torch.nn.Module):
    """
    Behler-style cosine cutoff module.
    [source https://github.com/atomistic-machine-learning/schnetpack/blob/
        master/src/schnetpack/nn/cutoff.py]

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)
          \right] & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Parameters
    ----------
    cutoff: float
        Cutoff distance
    width: float
        Cutoff width defining cutoff range (cutoff - width) to (cutoff)
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        cutoff: float,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize cutoff function.
        """

        super(CosineCutoff, self).__init__()

        # Set cutoff value in the register for model parameters
        self.register_buffer("cutoff", torch.tensor([cutoff], dtype=dtype))

        return
        
    def forward(
        self,
        distance: torch.Tensor
    ) -> torch.Tensor:

        """
        Forward pass of the cutoff function.

        Parameters
        ----------
        distance : torch.Tensor
            Atom pair distance tensor

        Returns
        -------
        torch.Tensor
            Cutoff values of input distance tensor

        """
        
        x = (distance - self.cutoff + self.width)/self.width
        
        cutoff = torch.where(
            distance < self.cutoff,
            torch.ones_like(distance),
            torch.zeros_like(distance)
            )

        cutoff = torch.where(
            torch.logical_and(
                (distance > self.cutoff - self.width),
                (distance < self.cutoff)
            ),
            0.5*(torch.cos(x*math.pi) + 1.0),
            cutoff)

        return cutoff

#======================================
# Function assignment
#======================================

functions_avaiable = {
    'default'.lower(): Poly6Cutoff,
    'Poly6'.lower(): Poly6Cutoff,
    'Poly6_Width'.lower(): Poly6Cutoff_Width,
    'Cosine'.lower(): CosineCutoff,
    'Cosine_Width'.lower(): CosineCutoff_Width,
    }


def get_cutoff_fn(
    name: Optional[Union[Callable, str]] = None,
) -> torch.nn.Module:
    """
    Get cutoff function by defined name.

    Parameters
    ----------
    name: (str, object), optional, default None
        If name is a str than it checks for the corresponding cutoff function
        and return the function object.
        The input will be given if it is already a function object.
        If None, then default cutoff function is used.

    Returns
    -------
    torch.nn.Module
        Cutoff function layer

    """

    # Check for default option
    if name is None:
        name = 'default'

    # Get cutoff function
    elif utils.is_callable(name):
        return name

    elif utils.is_string(name):
        if name.lower() in functions_avaiable:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Cutoff function input '{name}' is not valid! "
                + "Choose from:\n"
                + str(functions_avaiable.keys()))

    else:
        raise ValueError(
            f"Cutoff function input of type '{type(name)}' "
            + "is not valid! Input 'name' has to be an object or 'str' from;\n"
            + str(functions_avaiable.keys()))
