
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch
# import pytorch_lightning as pl

from .. import settings
from .. import utils
from .. import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_graph_model', 'Graph_PhysNetMP']

#======================================
# Graph Models
#======================================

class Graph_PhysNetMP(torch.nn.Module): 
    """
    Name: PhysNet Message Passing - Graph model

    Parameters
    ----------

    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_n_residual_interaction: int, optional, default 3
        Number of residual layers for message refinement
    graph_n_residual_atomic: int, optional, default 2
        Number of residual layers for atomic feature refinement
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    **kwargs: dict, optional
        Additional arguments

    Returns
    -------
    callable object
        PhysNet Message Passing graph model object
    """
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        graph_n_blocks: Optional[int] = None,
        graph_n_residual_interaction: Optional[int] = None,
        graph_n_residual_atomic: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        **kwargs
    ):
        """
        Initialize NNP graph model.

        """
        
        super().__init__()
        
        # Get configuration object
        config = settings.get_config(config)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'kwargs', '__class__']:
                continue

            # Take argument from global configuration dictionary if not defined
            # directly
            if item is None:
                item = config.get(arg)

            # Set default value if the argument is not defined (None)
            if arg in settings._default_args.keys() and item is None:
                item = settings._default_args[arg]

            # Check datatype of defined arguments
            if arg in settings._dtypes_args.keys():
                match = utils.check_input_dtype(
                    arg, item, settings._dtypes_args, raise_error=True)

            # Append to update dictionary
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)
            
        # Update global configuration dictionary
        config.update(config_update)
        
        # Graph class type
        self.graph_type = 'PhysNetMP'
        
        # Initialize activation function
        self.graph_activation_fn = layers.get_activation_fn(
            self.graph_activation_fn)
        
        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        # Get graph model interface parameters 
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.input_n_radialbasis = config.get('input_n_radialbasis')
        
        # Assign graph model training parameters
        self.rate = settings._global_rate
        
        self.interaction_blocks = torch.nn.ModuleList([
            layers.InteractionBlock(
                self.input_n_atombasis, 
                self.input_n_radialbasis, 
                self.graph_n_residual_atomic,
                self.graph_n_residual_interaction,
                self.graph_activation_fn,
                rate=self.rate,
                device=settings._global_device)
            for _ in range(self.graph_n_blocks)
            ])


    def forward(
        self, 
        features: torch.Tensor,
        descriptors: torch.Tensor, 
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> List[torch.Tensor]:

        """
        Forward pass of the graph model.
        Parameters
        ----------
        features : torch.Tensor
            Atomic feature vector
        descriptors : torch.Tensor
            Descriptor vector RBF
        idx_i: torch.Tensor
            Index of atom i
        idx_j: torch.Tensor
            Index of atom j

        Returns
        -------

        """
        #TODO: x should change after each interaction block(?)
        
        # Assign first atomic feature vector as message vector
        x = features
        
        # Initialize refined feature vector list
        x_all = []
        
        # Apply message passing model
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, descriptors, idx_i, idx_j)
            x_all.append(x)
        
        return x_all
    
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'graph_n_blocks': self.graph_n_blocks,
            'graph_n_residual_interaction': self.graph_n_residual_interaction,
            'graph_n_residual_atomic': self.graph_n_residual_atomic,
            }


class Graph_PaiNNMP(torch.nn.Module): 
    """
    Name: PaiNN Message Passing - Graph model

    Parameters
    ----------

    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    graph_stability_constant: float, optional, default 1.e-8
        Numerical stability constant added to scalar products of Cartesian 
        information vectors (guaranteed to be non-zero).
    **kwargs: dict, optional
        Additional arguments

    Returns
    -------
    callable object
        PhysNet Message Passing graph model object
    """
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        graph_n_blocks: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        graph_stability_constant: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize NNP graph model.

        """
        
        super().__init__()
        
        # Get configuration object
        config = settings.get_config(config)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = {}
        for arg, item in locals().items():

            # Skip 'config' argument and possibly more
            if arg in [
                    'self', 'config', 'config_update', 'kwargs', '__class__']:
                continue

            # Take argument from global configuration dictionary if not defined
            # directly
            if item is None:
                item = config.get(arg)

            # Set default value if the argument is not defined (None)
            if arg in settings._default_args.keys() and item is None:
                item = settings._default_args[arg]

            # Check datatype of defined arguments
            if arg in settings._dtypes_args.keys():
                match = utils.check_input_dtype(
                    arg, item, settings._dtypes_args, raise_error=True)

            # Append to update dictionary
            config_update[arg] = item

            # Assign as class parameter
            setattr(self, arg, item)
            
        # Update global configuration dictionary
        config.update(config_update)
        
        # Graph class type
        self.graph_type = 'PaiNNMP'
        
        # Initialize activation function
        self.graph_activation_fn = layers.get_activation_fn(
            self.graph_activation_fn)
        
        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        # Get graph model interface parameters 
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.input_n_radialbasis = config.get('input_n_radialbasis')
        
        # Assign graph model training parameters
        self.rate = settings._global_rate
        
        # Initialize feature-wise, continuous-filter convolution network
        self.descriptors_filter = layers.DenseLayer(
            Nin=self.input_n_radialbasis,
            Nout=self.graph_n_blocks*self.input_n_atombasis*3,
            activation_fn=None,
            W_init=True,
            bias=True,
            device=self.device,
            dtype=self.dtype)
        
        # Initialize message passing blocks
        self.interaction_block = torch.nn.ModuleList([
            layers.PaiNNInteraction(
                self.input_n_atombasis, 
                activation_fn=self.graph_activation_fn,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])
        self.mixing_block = torch.nn.ModuleList([
            layers.PaiNNMixing(
                self.input_n_atombasis, 
                activation_fn=self.graph_activation_fn,
                stability_constant=self.graph_stability_constant,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])

    def forward(
        self, 
        features: torch.Tensor,
        rbfs: torch.Tensor,
        distances: torch.Tensor,
        vectors: torch.Tensor,
        cutoff: torch.Tensor,
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> List[torch.Tensor]:

        """
        Forward pass of the graph model.
        Parameters
        ----------
        features : torch.Tensor
            Atomic feature vector
        rbfs : torch.Tensor
            Atom pair radial basis functions vector RBF
        distances : torch.Tensor
            Atom pair distances
        vectors : torch.Tensor
            Atom pair connection vectors
        cutoff : torch.Tensor
            Atom pair cutoff values with respect to distances
        idx_i: torch.Tensor
            Atom pair indices of atom i
        idx_j: torch.Tensor
            Atom pair indices of atom j

        Returns
        -------

        """
        
        # Apply feature-wise, continuous-filter convolution
        descriptors = (
            self.descriptors_filter(rbfs[:, None, :])*cutoff[:, None, None])
        descriptors_list = torch.split(
            descriptors, 3*self.input_n_atombasis, dim=-1)
        
        # Normalize atom pair vectors
        vectors_normalized = vectors/distances[:, None]

        # Assign isolated atomic feature vectors as scalar feature 
        # vectors
        fsize = features.shape # (len(atomic_numbers), input_n_atombasis)
        sfeatures = features[:, None, :]

        # Initialize equivariant feature vectors
        efeatures = torch.zeros((fsize[0], 3, fsize[1]), device=self.device)

        # Apply message passing model to modify from isolated atomic features
        # vectors to molecular atomic features as a function of the chemical
        # environment
        for ii, (interaction, mixing) in enumerate(
            zip(self.interaction_block, self.mixing_block)
        ):
            sfeatures, efeatures = interaction(
                sfeatures, 
                efeatures, 
                descriptors_list[ii], 
                vectors_normalized, 
                idx_i, 
                idx_j,
                fsize[0], 
                fsize[1])
            sfeatures, efeatures = mixing(
                sfeatures, 
                efeatures, 
                fsize[1])
        
        # Flatten scalar atomic feature vector
        sfeatures = sfeatures.squeeze(1)
        
        return sfeatures, efeatures
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {
            'graph_n_blocks': self.graph_n_blocks,
            'graph_stability_constant': self.graph_stability_constant,
            }


#======================================
# Graph Model Assignment  
#======================================

graph_model_available = {
    'PhysNetMP'.lower(): Graph_PhysNetMP,
    'PaiNNMP'.lower(): Graph_PaiNNMP,
    }

def get_graph_model(
    config: Optional[Union[str, dict, object]] = None,
    graph_type: Optional[str] = None,
    **kwargs
):
    """
    Input module selection
    
    Parameters
    ----------
    
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    graph_type: str, optional, default 'RBF'
        Graph model representation of the information processing
        e.g. 'PhysNetMP'
    **kwargs: dict, optional
        Additional arguments for parameter initialization
    
    Returns
    -------
    callable object
        Graph model object to process atomistic and structural information
    """
    
    # Get configuration object
    config = settings.get_config(config)
    
    # Check input parameter, set default values if necessary and
    # update the configuration dictionary
    config_update = {}
    for arg, item in locals().items():
        
        # Skip 'config' argument and possibly more
        if arg in ['self', 'config', 'config_update', 'kwargs', '__class__']:
            continue
        
        # Take argument from global configuration dictionary if not defined
        # directly
        if item is None:
            item = config.get(arg)
        
        # Set default value if the argument is not defined (None)
        if arg in settings._default_args.keys() and item is None:
            item = settings._default_args[arg]
        
        # Check datatype of defined arguments
        if arg in settings._dtypes_args.keys():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)
        
        # Append to update dictionary
        config_update[arg] = item
        
    # Update global configuration dictionary
    config.update(config_update)

    # Check graph model type
    if config.get('graph_type') is None:
        model_type = config.get('model_type')
        if settings._available_graph_model.get(model_type) is None:
            raise SyntaxError(
                "No graph model type could assigned from defined model "
                + f"type '{model_type:s}'!")
        config['graph_type'] = settings._available_graph_model.get(model_type)
    graph_type = config['graph_type']

    # Graph model type assignment
    if (
        graph_type.lower() in 
        [key.lower() for key in graph_model_available.keys()]
    ):
        return graph_model_available[graph_type.lower()](
            config,
            **kwargs)
    else:
        raise ValueError(
            f"Graph model type graph '{graph_type:s}' is not valid!\n" +
            "Choose from:\n" + str(graph_model_available.keys()))
    
    return
