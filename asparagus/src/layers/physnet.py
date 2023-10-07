
from typing import Optional, List, Dict, Tuple, Union

import torch

from .. import settings
from .. import utils

class InteractionBlock(torch.nn.Module):

    def __init__(
        self,
        input_n_atombasis: int,
        input_n_radialbasis: int,
        graph_n_residual_atomic: int,
        graph_n_residual_interaction: int,
        activation_fn: Optional[object] = None,
        rate: Optional[float] = 0.0,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(InteractionBlock, self).__init__()

        # Interaction Layer
        self.interaction = InteractionLayer(
            input_n_atombasis,
            input_n_radialbasis,
            graph_n_residual_interaction,
            activation_fn,
            rate=rate,
            device=device,
            dtype=dtype)

        # Residual Layers
        self.residual_layers = torch.nn.ModuleList([
            ResidualLayer(
                input_n_atombasis,
                input_n_atombasis,
                activation_fn=activation_fn,
                rate=rate,
                device=device,
                dtype=dtype)
            for _ in range(graph_n_residual_atomic)])

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:

        # Assign first atomic feature vector as message vector
        x = features

        # Apply interaction layer
        x = self.interaction(x, descriptors, idx_i, idx_j)

        # Iterate through residual layers
        for residual_layer in self.residual_layers:
            x = residual_layer(x)

        return x


class InteractionLayer(torch.nn.Module):

    def __init__(
        self,
        input_n_atombasis: int,
        input_n_radialbasis: int,
        graph_n_residual_interaction: int,
        activation_fn: Optional[object] = None,
        rate: Optional[float] = 0.0,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(InteractionLayer, self).__init__()

        # Assign activation function
        # self.activation_fn = activation_fn
        if activation_fn is None:
            self.use_activation = False
        else:
            self.activation_fn = activation_fn
            self.use_activation = True

        # Assign device for utils.segment_sum
        self.device = device

        # Dropout layer
        if settings._global_mode == 'train' or rate > 0.0:
            self.use_dropout = True
            self.dropout = torch.nn.Dropout(rate)
        else:
            self.use_dropout = False

        # Dense layers
        self.desc2feat = DenseLayer(
            input_n_radialbasis,
            input_n_atombasis,
            W_init=False,
            bias=False,
            device=device,
            dtype=dtype)
        self.dense_i = DenseLayer(
            input_n_atombasis,
            input_n_atombasis,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype)
        self.dense_j = DenseLayer(
            input_n_atombasis,
            input_n_atombasis,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype)

        # Residual layers
        self.residual_layers = torch.nn.ModuleList([
            ResidualLayer(
                input_n_atombasis,
                input_n_atombasis,
                activation_fn=activation_fn,
                rate=rate,
                device=device,
                dtype=dtype)
            for _ in range(graph_n_residual_interaction)])

        # For performing the final update to the feature vectors
        self.dense = DenseLayer(
            input_n_atombasis,
            input_n_atombasis,
            device=device,
            dtype=dtype)
        self.u = torch.nn.Parameter(
            torch.ones([input_n_atombasis], device=device, dtype=dtype))

    def forward(
        self,
        features: torch.Tensor,
        descriptors: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:

        # Assign atomic feature vector as message vector
        x = features

        # Apply Pre-activation #TODO: Check if this is efficient
        # if self.use_dropout:
        #     xa = self.dropout(self.activation_fn(x))
        # else:
        #     xa = self.activation_fn(x)
        # Apply Pre-activation
        if self.use_activation and self.use_dropout:
            xa = self.dropout(self.activation_fn(x))
        elif self.use_activation:
            xa = self.activation_fn(x)
        elif self.use_dropout:
            xa = self.dropout(x)
        else:
            xa = x

        # Calculate feature mask from radial basis functions
        g = self.desc2feat(descriptors)

        # Calculate contribution of neighbors and central atom
        xi = self.dense_i(xa)
        if self.device == 'cpu':
            pxj = g*self.dense_j(xa)[idx_j]
        else:
            j = idx_j.view(-1, 1).expand(-1, x.shape[-1])
            pxj = g * torch.gather(self.dense_j(xa), 0, j)

        xj = utils.segment_sum(pxj, idx_i, device=self.device)

        # Sum of messages
        message = xi + xj

        # Apply residual layers
        for residual_layer in self.residual_layers:
            message = residual_layer(message)

        message = self.activation_fn(message)

        x = self.u*x + self.dense(message)

        return x


class ResidualLayer(torch.nn.Module):

    def __init__(
        self,
        Nin: int,
        Nout: int,
        activation_fn: Optional[object] = None,
        rate: Optional[float] = 0.0,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(ResidualLayer, self).__init__()

        # Dropout layer
        if settings._global_mode == 'train' or rate > 0.0:
            self.use_dropout = True
            self.dropout = torch.nn.Dropout(rate)
        else:
            self.use_dropout = False

        # Assign activation function
        if activation_fn is None:
            self.use_activation = False
        else:
            self.activation_fn = activation_fn
            self.use_activation = True

        self.dense = DenseLayer(
            Nin,
            Nout,
            activation_fn=activation_fn,
            device=device,
            dtype=dtype)
        self.residual = DenseLayer(
            Nout,
            Nout,
            device=device,
            dtype=dtype)

    def forward(
        self,
        message: torch.Tensor,
    ) -> torch.Tensor:

        if self.use_activation and self.use_dropout:
            y = self.dropout(self.activation_fn(message))
        elif self.use_activation:
            y = self.activation_fn(message)
        elif self.use_dropout:
            y = self.dropout(message)
        else:
            y = message

        # Apply residual layer
        message = message + self.residual(self.dense(y))

        return message


class OutputBlock(torch.nn.Module):

    def __init__(
        self,
        input_n_atombasis: int,
        output_n_residual: int,
        activation_fn: Optional[object] = None,
        output_n_results: Optional[int] = 1,
        rate: Optional[float] = 0.0,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(OutputBlock, self).__init__()

        # Assign activation function
        self.activation_fn = activation_fn

        # Dropout layer
        if settings._global_mode == 'train' or rate > 0.0:
            self.use_dropout = True
            self.dropout = torch.nn.Dropout(rate)
        else:
            self.use_dropout = False

        # Residual layers
        self.residual_layers = torch.nn.ModuleList([
            ResidualLayer(
                input_n_atombasis,
                input_n_atombasis,
                activation_fn=activation_fn,
                rate=rate,
                device=device,
                dtype=dtype)
            for _ in range(output_n_residual)])

        # Output
        self.dense = DenseLayer(
            input_n_atombasis,
            output_n_results,
            activation_fn=activation_fn,
            W_init=False,
            bias=False,
            device=device,
            dtype=dtype)

    def forward(
        self,
        feature: torch.Tensor
    ) -> torch.Tensor:

        # Apply residual layers
        for residual_layer in self.residual_layers:
            feature = residual_layer(feature)

        # Apply activation function
        if self.activation_fn is not None:
            feature = self.activation_fn(feature)

        # Get output value(s)
        output = self.dense(feature)

        return output


class DenseLayer(torch.nn.Module):

    def __init__(
        self,
        Nin: int,
        Nout: int,
        activation_fn: Optional[object] = None,
        W_init: Optional[bool] = True,
        bias: Optional[bool] = True,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):

        super(DenseLayer, self).__init__()

        # Assign activation function
        if activation_fn is None:
            self.use_activation = False
        else:
            self.activation_fn = activation_fn
            self.use_activation = True

        # Initialize Linear NN layer
        self.linear = torch.nn.Linear(
            Nin, Nout, bias=bias, device=device, dtype=dtype)

        # Initialize NN weights
        if W_init:
            torch.nn.init.xavier_normal_(self.linear.weight, gain=1.e-4)
        else:
            torch.nn.init.zeros_(self.linear.weight)

        # Initialize NN bias
        if bias:
            torch.nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        message: torch.Tensor,
    ) -> torch.Tensor:

        if self.use_activation:
            return self.activation_fn(self.linear(message))
        else:
            return self.linear(message)
