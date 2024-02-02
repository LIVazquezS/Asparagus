import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

__all__ = [
    'get_radial_fn', 'RBF_PhysNet', 'RBF_PhysNet_original'
    ]


#======================================
# Radial basis functions
#======================================

class RBF_PhysNet(torch.nn.Module):
    """
    PhysNet type radial basis functions (RBFs).


    Parameters
    ----------
    rbf_n_basis: int
        Number of RBFs
    rbf_cutoff_fn: object
        Cutoff function
    rbf_center_start: float
        Initial lower RBF center range
    rbf_center_end: float
        Initial upper RBF center range
    rbf_trainable: bool
        Trainable RBF center positions

    Returns
    -------
    object
        Input model object to encode atomistic structural information

    """

    def __init__(
        self,
        rbf_n_basis: int,
        rbf_cutoff_fn: object,
        rbf_center_start: float,
        rbf_center_end: float,
        rbf_trainable: bool,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        NNP Input Module

        """

        super(RBF_PhysNet, self).__init__()

        self.rbf_n_basis = rbf_n_basis
        self.rbf_cutoff_fn = rbf_cutoff_fn
        self.rbf_trainable = rbf_trainable

        # Initialize RBF centers and widths
        centers = torch.linspace(
            rbf_center_start,
            rbf_center_end,
            rbf_n_basis,
            device=device,
            dtype=dtype)
        # TODO check if still broken or width can be estimated this way
        # Update: it seems to work, still unsure if good initial guess.
        widths = (
            (0.5*rbf_n_basis/(rbf_center_start - rbf_center_end))**2
            * torch.ones(rbf_n_basis, device=device, dtype=dtype))
        #widths = torch.ones_like(centers, device=device, dtype=dtype)

        if rbf_trainable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    @staticmethod
    def softplus_inverse(x):
        '''
        Inverse softplus function

        .. math:: f(x) = ln(exp(x) - 1)

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------

        '''
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return torch.log(torch.exp(x) - 1)

    def forward(
        self,
        d: torch.Tensor
    ) -> torch.Tensor:

        x = torch.unsqueeze(d, -1)
        rbf = self.rbf_cutoff_fn(x)*torch.exp(
            -self.widths*(x - self.centers)**2)

        return rbf


class RBF_PhysNet_original(torch.nn.Module):
    """
    Original PhysNet type radial basis functions (RBFs) with double exponential
    expression.


    Parameters
    ----------
    rbf_n_basis: int
        Number of RBFs
    rbf_cutoff_fn: object
        Cutoff function
    rbf_center_start: float
        Initial lower RBF center range
    rbf_center_end: float
        Initial upper RBF center range
    rbf_trainable: bool
        Trainable RBF center positions

    Returns
    -------
    object
        Input model object to encode atomistic structural information

    """

    def __init__(
        self,
        rbf_n_basis: int,
        rbf_cutoff_fn: object,
        rbf_center_start: float,
        rbf_center_end: float,
        rbf_trainable: bool,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        NNP Input Module

        """

        super(RBF_PhysNet_original, self).__init__()

        self.rbf_n_basis = rbf_n_basis
        self.rbf_cutoff_fn = rbf_cutoff_fn
        self.rbf_trainable = rbf_trainable

        # Initialize RBF centers and widths
        centers = torch.linspace(
            rbf_center_start, np.exp(-rbf_center_end), rbf_n_basis)
        softp = torch.nn.Softplus()
        width_val = self.softplus_inverse(
            (0.5 / ((1.0 - np.exp(-rbf_center_end)) / rbf_n_basis)) ** 2)
        widths = torch.empty(
            rbf_n_basis, dtype=torch.float64).new_full(
                (rbf_n_basis,), softp(width_val * rbf_n_basis))

        if rbf_trainable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    @staticmethod
    def softplus_inverse(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return torch.log(torch.exp(x) - 1)

    def forward(
        self,
        d: torch.Tensor
    ) -> torch.Tensor:

        x = torch.unsqueeze(d, -1)
        rbf = self.rbf_cutoff_fn(x)*torch.exp(
            -self.widths*(torch.exp(-x) - self.centers)**2)

        return rbf


class RBF_PaiNN(torch.nn.Module):
    """
    PaiNN type radial basis functions (RBFs).


    Parameters
    ----------
    rbf_n_basis: int
        Number of RBFs
    rbf_cutoff_fn: object
        Cutoff function
    rbf_center_start: float
        Initial lower RBF center range
    rbf_center_end: float
        Initial upper RBF center range
    rbf_trainable: bool
        Trainable RBF center positions

    Returns
    -------
    object
        Input model object to encode atomistic structural information

    """

    def __init__(
        self,
        rbf_n_basis: int,
        rbf_cutoff_fn: object,
        rbf_center_start: float,
        rbf_center_end: float,
        rbf_trainable: bool,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        NNP Input Module

        """

        super(RBF_PaiNN, self).__init__()

        self.rbf_n_basis = rbf_n_basis
        self.rbf_cutoff_fn = rbf_cutoff_fn
        self.rbf_trainable = rbf_trainable

        # Initialize RBF centers and widths
        centers = torch.linspace(
            rbf_center_start,
            rbf_center_end,
            rbf_n_basis,
            device=device,
            dtype=dtype)
        widths = (
            torch.abs(centers[1] - centers[0])
            * torch.ones_like(centers, device=device, dtype=dtype))

        if rbf_trainable:
            self.centers = torch.nn.Parameter(centers)
            self.widths = torch.nn.Parameter(widths)
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("widths", widths)

    def forward(
        self,
        d: torch.Tensor
    ) -> torch.Tensor:

        cutoff = self.rbf_cutoff_fn(d)
        rbfs = torch.exp(
            -0.5*((torch.unsqueeze(d, -1) - self.centers)/self.widths)**2)

        return rbfs, cutoff


#======================================
# Function assignment
#======================================

functions_avaiable = {
    'RBF_PhysNet'.lower(): RBF_PhysNet,
    'RBF_PaiNN'.lower(): RBF_PaiNN,
    'RBF_PhysNet_original'.lower(): RBF_PhysNet_original,
    }


def get_radial_fn(name):
    """
    Get radial basis function by defined name.

    Parameters
    ----------
    name: (str, object)
        If name is a str than it checks for the corresponding radial basis
        function and return the function object.
        The input will be given if it is already a function object.

    Returns
    -------
    callable
        Radial basis function
    """

    if utils.is_callable(name):

        return name

    if utils.is_string(name):

        if name.lower() in [key.lower() for key in functions_avaiable.keys()]:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Radial basis function input '{name}' is not valid!" +
                "Choose from:\n" +
                str(functions_avaiable.keys()))
    else:

        raise ValueError(
            f"Radial basis function input of type '{type(name)}' " +
            "is not valid! Input 'name' has to be an object or 'str' from;\n" +
            str(functions_avaiable.keys()))
