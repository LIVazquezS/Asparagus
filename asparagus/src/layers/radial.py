import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from .. import utils

__all__ = [
    'get_radial_fn', 'PhysNetRBF'
    ]


#======================================
# Radial basis functions
#======================================

class PhysNetRBF(torch.nn.Module):
    """
    PhysNet type radial basis functions (RBFs).
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
        
        super(PhysNetRBF, self).__init__()
        
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
        #widths = torch.tensor(
            #(0.5*rbf_n_basis/(centers[0] - centers[-1]))**2
            #*torch.ones_like(centers),
            #device=device, dtype=dtype)
        widths = torch.ones_like(centers, device=device, dtype=dtype)
        
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

        x = torch.unsqueeze(d, -1) #d[..., None]
        rbf = self.rbf_cutoff_fn(x)*torch.exp(
            -self.widths*(x - self.centers)**2)
        
        return rbf



class PhysNetRBF_STUPIDDDDDDDD(torch.nn.Module):
    """
    STUPIDDDDDDDD PhysNet type radial basis functions (RBFs).
    This need to be incorporated into the PhysNetRBF class.
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
        
        super(PhysNetRBF_STUPIDDDDDDDD, self).__init__()
        
        self.rbf_n_basis = rbf_n_basis
        self.rbf_cutoff_fn = rbf_cutoff_fn
        self.rbf_trainable = rbf_trainable
        
        # Initialize RBF centers and widths
        centers = torch.linspace(rbf_center_start, np.exp(-rbf_center_end), rbf_n_basis)
        softp = torch.nn.Softplus()
        width_val = self.softplus_inverse((0.5 / ((1.0 - np.exp(-rbf_center_end)) / rbf_n_basis)) ** 2)
        widths = torch.empty(rbf_n_basis, dtype=torch.float64).new_full((rbf_n_basis,), softp(width_val * rbf_n_basis))

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

        x = torch.unsqueeze(d, -1) #d[..., None]
        rbf = self.rbf_cutoff_fn(x)*torch.exp(
            -self.widths*(torch.exp(-x) - self.centers)**2)
        
        return rbf

#======================================
# Function assignment  
#======================================

functions_avaiable = {
    'PhysNetRBF'.lower(): PhysNetRBF,
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
        object
            Radial basis function
    """
    
    if utils.is_object(name):
        
        return name
    
    if utils.is_string(name):
        
        if name.lower() in [key.lower() for key in functions_avaiable.keys()]:
            return functions_avaiable[name.lower()]
        else:
            raise ValueError(
                f"Radial basis function input '{name}' is not valid!" +
                f"Choose from:\n" +
                str(functions_avaiable.keys()))
    else:
        
        raise ValueError(
            f"Radial basis function input of type '{type(name)}' " +
            f"is not valid! Input 'name' has to be an object or 'str' from;\n" +
            str(functions_avaiable.keys()))