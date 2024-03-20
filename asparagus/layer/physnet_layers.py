from typing import Optional, Callable

import torch

from .base import DenseLayer, ResidualLayer

from .. import utils

__all__ = ['InteractionBlock', 'InteractionLayer', 'OutputBlock']

#======================================
# PhysNet NN Blocks
#======================================


class InteractionBlock(torch.nn.Module):
    """
    Interaction block for PhysNet

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_radialbasis: int
        Number of input radial basis centers
    n_residual_interaction: int
        Number of residual layers for atomic feature and radial basis vector
        interaction.
    n_residual_features: int
        Number of residual layers for atomic feature interactions.
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
        n_radialbasis: int,
        n_residual_interaction: int,
        n_residual_features: int,
        activation_fn: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        """
        Initialize PhysNet interaction block.
        """

        super(InteractionBlock, self).__init__()

        # Atomic features and radial basis vector interaction layer
        self.interaction = InteractionLayer(
            n_atombasis,
            n_radialbasis,
            n_residual_interaction,
            activation_fn,
            device=device,
            dtype=dtype)

        # Atomic feature interaction layers
        self.residuals = torch.nn.ModuleList([
            ResidualLayer(
                n_atombasis,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype)
            for _ in range(n_residual_features)])

        return

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply interaction block.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        descriptors: torch.Tensor(N_atoms, n_atombasis, n_radialbasis)
            Atom pair radial distribution vectors
        idx_i: torch.Tensor(N_pairs)
            Atom i pair index
        idx_j: torch.Tensor(N_pairs)
            Atom j pair index

        Returns
        -------
        torch.Tensor(N_atoms, n_atombasis)
            Modified atom feature vectors
        """

        # Apply interaction layer
        x = self.interaction(features, descriptors, idx_i, idx_j)

        # Iterate through atomic feature interaction layers
        for residual in self.residuals:
            x = residual(x)

        return x


class InteractionLayer(torch.nn.Module):
    """
    Atomic features and radial basis vector interaction layer for PhysNet

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_radialbasis: int
        Number of input radial basis centers
    n_residual_interaction: int
        Number of residual layers for atomic feature and radial basis vector
        interaction.
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
        n_radialbasis: int,
        n_residual_interaction: int,
        activation_fn: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(InteractionLayer, self).__init__()

        # Assign activation function
        if activation_fn is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation_fn

        # Dense layer for the conversion from radial basis vector to atomic 
        # feature vector length
        self.radial2atom = DenseLayer(
            n_radialbasis,
            n_atombasis,
            bias=False,
            weight_init=torch.nn.init.zeros_,
            device=device,
            dtype=dtype)

        # Dense layer for atomic feature vector for atom i
        self.dense_i = DenseLayer(
            n_atombasis,
            n_atombasis,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype)
        
        # Dense layer for atomic feature vector for atom j
        self.dense_j = DenseLayer(
            n_atombasis,
            n_atombasis,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype)

        # Residual layers for atomic feature vector pair interaction modifying 
        # the message vector
        self.residuals_ij = torch.nn.ModuleList([
            ResidualLayer(
                n_atombasis,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype)
            for _ in range(n_residual_interaction)])

        # Dense layer for message vector interaction
        self.dense_out = DenseLayer(
            n_atombasis,
            n_atombasis,
            device=device,
            dtype=dtype)
        
        # Scaling vector for mixing of initial atomic feature vector with
        # message vector
        self.scaling = torch.nn.Parameter(
            torch.ones([n_atombasis], device=device, dtype=dtype))
        
        # Special case flag for variable assignment on CPU's
        if device.lower() == 'cpu':
            self.cpu = True
        else:
            self.cpu = False

        return

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply interaction layer.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        descriptors: torch.Tensor(N_atoms, n_atombasis, n_radialbasis)
            Atom pair radial distribution vectors
        idx_i: torch.Tensor(N_pairs)
            Atom i pair index
        idx_j: torch.Tensor(N_pairs)
            Atom j pair index

        Returns
        -------
        torch.Tensor(N_atoms, n_atombasis)
            Modified atom feature vectors
        
        """

        # Apply initial activation function on atomic features
        x = self.activation(features)

        # Apply radial basis (descriptor) to feature vector layer
        g = self.radial2atom(descriptors)

        # Calculate contribution of central atom i and neighbor atoms j
        xi = self.dense_i(x)
        if self.cpu:
            gxj = g*self.dense_j(x)[idx_j]
        else:
            j = idx_j.view(-1, 1).expand(-1, features.shape[-1])
            gxj = g * torch.gather(self.dense_j(x), 0, j)

        # Combine descriptor weighted neighbor atoms feature vector for each
        # central atom i
        xj = utils.segment_sum(gxj, idx_i, device=gxj.device)

        # Combine features to message vector
        message = xi + xj

        # Apply residual layers and acitvation function for message vector
        # interaction
        for residual in self.residuals_ij:
            message = residual(message)
        message = self.activation(message)

        # Mix initial atomic feature vector with message vector
        x = self.scaling*x + self.dense_out(message)

        return x


class OutputBlock(torch.nn.Module):
    """
    Output block for PhysNet

    Parameters
    ----------
    n_atombasis: int
        Number of atomic features (length of the atomic feature vector)
    n_results: int
        Number of output vector features.
    n_residual: int
        Number of residual layers for transformation from atomic feature vector
        to output results.
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
        n_results: int,
        n_residual: int,
        activation_fn: Optional[Callable] = None,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(OutputBlock, self).__init__()

        # Assign activation function
        if activation_fn is None:
            self.activation_fn = torch.nn.Identity()
        else:
            self.activation_fn = activation_fn

        # Residual layer for atomic feature modification
        self.residuals = torch.nn.ModuleList([
            ResidualLayer(
                n_atombasis,
                activation_fn=activation_fn,
                device=device,
                dtype=dtype)
            for _ in range(n_residual)])

        # Dense layer for transforming atomic feature vector to result vector
        self.output = DenseLayer(
            n_atombasis,
            n_results,
            activation_fn=activation_fn,
            bias=False,
            weight_init=torch.nn.init.zeros_,
            device=device,
            dtype=dtype)
        
        return

    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply output block.
        
        Parameters
        ----------
        features: torch.Tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        
        Returns
        -------
        torch.Tensor(N_atoms, n_results)
            Transformed atomic feature vector to result vector
        
        """

        # Apply residual layers on atomic features
        for residual in self.residuals:
            features = residual(features)

        # Apply last activation function
        features = self.activation_fn(features)

        # Transform to result vector
        result = self.output(features)

        return result
