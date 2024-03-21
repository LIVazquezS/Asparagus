from typing import Optional, List, Union, Callable

import torch

from .base import DenseLayer

from .. import utils

__all__ = ['PaiNNInteraction', 'PaiNNMixing', 'PaiNNOutput']

class PaiNNInteraction(torch.nn.Module):
    """
    Interaction block for PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    activation_fn: callable, optional, default None
        Residual layer activation function.
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """

    def __init__(
        self,
        n_atombasis: int,
        activation_fn: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN interaction block.
        
        """

        super(PaiNNInteraction, self).__init__()

        # Initialize context layer
        self.context = torch.nn.Sequential(
            DenseLayer(
                n_atombasis, 
                n_atombasis,
                bias=True,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            DenseLayer(
                n_atombasis, 
                3*n_atombasis,
                bias=True,
                activation_fn=None,
                device=device,
                dtype=dtype
                ),
            )

        return

    def forward(
        self,
        sfeatures: torch.Tensor,
        efeatures: torch.Tensor,
        descriptors: torch.Tensor,
        vectors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        n_atoms: int,
        n_features: int
    ) -> (torch.Tensor, torch.Tensor):

        # Apply context layer on scalar features
        cfea = self.context(sfeatures)

        # Apply conformational information on context of scalar features
        cfea = descriptors*cfea[idx_j]

        # Split update information
        ds, dev, dee = torch.split(
            cfea, n_features, dim=-1)

        # Reduce from pair information to atom i
        ds = torch.zeros(
            (n_atoms, 1, n_features),
            device=ds.device, dtype=ds.dtype
            ).index_add(0, idx_i, ds)

        # Compute equivariant feature vector update and reduce to atom i
        de = dev*vectors[..., None] + dee*efeatures[idx_j]
        de = torch.zeros(
            (n_atoms, 3, n_features),
            device=de.device, dtype=de.dtype
            ).index_add(0, idx_i, de)

        # Update scalar and equivariant feature vector
        sfeatures = sfeatures + ds
        efeatures = efeatures + de

        return sfeatures, efeatures


class PaiNNMixing(torch.nn.Module):
    """
    Mixing block of scalar and equivariant feature vectors in PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    activation_fn: callable, optional, default None
        Residual layer activation function.
    stability_constant: float, optional, default 1e-8
        Numerical stability added constant
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type
    
    """
    
    def __init__(
        self,
        n_atombasis: int,
        activation_fn: Optional[object] = None,
        stability_constant: Optional[float] = 1.e-8,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN mixing block.
        
        """

        super(PaiNNMixing, self).__init__()

        # Assign class parameter
        self.stability_constant = stability_constant
        
        # Initialize context and mixing layer
        self.context = DenseLayer(
            n_atombasis, 
            2*n_atombasis,
            activation_fn=activation_fn,
            bias=False,
            device=device,
            dtype=dtype
            )
        self.mixing = torch.nn.Sequential(
            DenseLayer(
                2*n_atombasis, 
                n_atombasis,
                bias=True,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            DenseLayer(
                n_atombasis, 
                3*n_atombasis,
                activation_fn=None,
                bias=False,
                device=device,
                dtype=dtype
                ),
            )

        return

    def forward(
        self,
        sfeatures: torch.Tensor,
        vfeatures: torch.Tensor,
        n_features: int,
    ) -> (torch.Tensor, torch.Tensor):
        
        # Apply context layer on vectorial feature vector and split 
        # information
        cfea = self.context(vfeatures)
        U, V = torch.split(cfea, n_features, dim=-1)
        
        # Mix scalar and vectorial feature vector
        U = torch.sqrt(
            torch.sum(U**2, dim=-2, keepdim=True) 
            + self.stability_constant)
        mix = torch.cat([sfeatures, U], dim=-1)
        mix = self.mixing(mix)
        
        # Split update information
        ds, dv, dsv = torch.split(mix, n_features, dim=-1)
        
        # Compute scalar and vectorial feature vector update
        ds = ds + dsv*torch.sum(U*V, dim=1, keepdim=True)
        dv = dv*_V
        
        # Update feature and message vector 
        sfeatures = sfeatures + ds
        vfeatures = vfeatures + dv
        
        return sfeatures, vfeatures


class PaiNNOutput(torch.nn.Module):
    """
    Output module for properties from representation of PaiNN.

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_properties: int
        ...
    n_hiddenlayers: int
        ...
    n_hiddenneurons: (int list(int)), optional, default None
        ...
    activation_fn: callable, optional, default None
        Residual layer activation function.
    stability_constant: float, optional, default 1e-8
        Numerical stability added constant
    device: str, optional, default 'cpu'
        Device type for model variable allocation
    dtype: dtype object, optional, default 'torch.float64'
        Model variables data type

    """
    def __init__(
        self,
        n_atombasis: int,
        n_properties: int,
        n_hiddenlayers: int,
        n_hiddenneurons: Optional[Union[int, List[int]]] = None,
        activation_fn: Optional[object] = None,
        last_bias: Optional[bool] = True,
        last_init_zero: Optional[bool] = False,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PaiNN output block.
        
        """
        
        super(PaiNNOutput, self).__init__()

        # Check hidden layer neuron option
        if n_hiddenlayers:
            if n_hiddenneurons is None:
                # Half number of hidden layer neurons with each layer
                n_neurons = n_atombasis
                n_hiddenneurons = []
                for ii in range(n_hiddenlayers):
                    n_hiddenneurons.append(n_neurons)
                    n_neurons = max(n_properties, n_neurons)
                n_hiddenneurons.append(n_properties)
            elif utils.is_integer(n_hiddenneurons):
                n_hiddenneurons = [n_hiddenneurons]*n_hiddenlayers
        else:
            # If no hidden layer, set hidden neurons to property neuron number
            n_hiddenneurons = [n_properties]

        # Initialize output module
        self.output = torch.nn.Sequential(
            DenseLayer(
                n_atombasis, 
                n_hiddenneurons[0],
                bias=True,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            )
        
        # Append hidden layers
        for ii in range(n_hiddenlayers):
            self.output.append(
                DenseLayer(
                    n_hiddenneurons[ii], 
                    n_hiddenneurons[ii + 1],
                    bias=True,
                    activation_fn=activation_fn,
                    device=device,
                    dtype=dtype
                    ),
                )
        
        # Append output layer
        self.output.append(
            DenseLayer(
                n_hiddenneurons[ii], 
                n_hiddenneurons[ii + 1],
                activation_fn=None,
                bias=last_bias,
                W_init=last_init_zero,
                device=device,
                dtype=dtype
                ),
            )
        
        return

    def forward(self):
        pass
    
