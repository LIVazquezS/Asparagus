import logging
import numpy as np
from typing import Optional, Union, List, Dict, Callable, Any

import torch

from .. import module
from .. import layer
from .. import settings
from .. import utils

from ..layer import painn_layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Input_PaiNN', 'Graph_PaiNN', 'Output_PaiNN']

#======================================
# Input Module
#======================================

class Input_PaiNN(torch.nn.Module):
    """
    PaiNN input module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    input_n_atombasis: int, optional, default 128
        Number of atomic features (length of the atomic feature vector)
    input_radial_fn: (str, callable), optional, default 'GaussianRBF'
        Type of the radial basis function.
    input_n_radialbasis: int, optional, default 64
        Number of input radial basis centers
    input_cutoff_fn: (str, callable), optional, default 'Poly6'
        Cutoff function type for radial basis function scaling
    input_radial_cutoff: float, optional, default 8.0
        Cutoff distance radial basis function
    input_rbf_center_start: float, optional, default 1.0
        Lowest radial basis center distance.
    input_rbf_center_end: float, optional, default None (input_radial_cutoff)
        Highest radial basis center distance. If None, radial basis cutoff
        distance is used.
    input_rbf_trainable: bool, optional, default True
        If True, radial basis function parameter are optimized during training.
        If False, radial basis function parameter are fixed.
    input_n_maxatom: int, optional, default 94 (Plutonium)
        Highest atom order number to initialize atom feature vector library.

    """

    # Default arguments for input module
    _default_args = {
        'input_n_atombasis':            128,
        'input_radial_fn':              'GaussianRBF',
        'input_n_radialbasis':          64,
        'input_cutoff_fn':              'Poly6',
        'input_radial_cutoff':          8.0,
        'input_rbf_center_start':       1.0,
        'input_rbf_center_end':         None,
        'input_rbf_trainable':          True,
        'input_n_maxatom':              94,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'input_n_atombasis':            [utils.is_integer],
        'input_radial_fn':              [utils.is_string, utils.is_callable],
        'input_n_radialbasis':          [utils.is_integer],
        'input_cutoff_fn':              [utils.is_string, utils.is_callable],
        'input_radial_cutoff':          [utils.is_numeric],
        'input_rbf_center_start':       [utils.is_numeric],
        'input_rbf_center_end':         [utils.is_None, utils.is_numeric],
        'input_rbf_trainable':          [utils.is_bool],
        'input_n_maxatom':              [utils.is_integer],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        input_n_atombasis: Optional[int] = None,
        input_radial_fn: Optional[Union[str, object]] = None,
        input_n_radialbasis: Optional[int] = None,
        input_cutoff_fn: Optional[Union[str, object]] = None,
        input_radial_cutoff: Optional[float] = None,
        input_rbf_center_start: Optional[float] = None,
        input_rbf_center_end: Optional[float] = None,
        input_rbf_trainable: Optional[bool] = None,
        input_n_maxatom: Optional[int] = None,
        input_atom_features_range: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize PaiNN input model.

        """

        super(Input_PaiNN, self).__init__()
        input_type = 'PaiNN'

        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )
            
        # Update global configuration dictionary
        config.update(config_update)
        
        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')
        
        # Check general model cutoff with radial basis cutoff
        if config.get('model_cutoff') is None:
            raise ValueError(
                "No general model interaction cutoff 'model_cutoff' is yet "
                + "defined for the model calculator!")
        elif config['model_cutoff'] < self.input_radial_cutoff:
            raise ValueError(
                "The model interaction cutoff distance 'model_cutoff' "
                + f"({self.model_cutoff:.2f}) must be larger or equal "
                + "the descriptor range 'input_radial_cutoff' "
                + f"({config.get('input_radial_cutoff'):.2f})!")

        ####################################
        # # # Input Module Class Setup # # #
        ####################################
        
        # Initialize atomic feature vectors
        self.atom_features = torch.nn.Embedding(
            self.input_n_maxatom + 1,
            self.input_n_atombasis, 
            padding_idx=0,
            device=self.device, 
            dtype=self.dtype)

        # Initialize radial cutoff function
        self.cutoff = layer.get_cutoff_fn(self.input_cutoff_fn)(
            self.input_radial_cutoff)
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_radial_cutoff
        
        # Initialize Radial basis function
        radial_fn = layer.get_radial_fn(self.input_radial_fn)
        self.radial_fn = radial_fn(
            self.input_n_radialbasis,
            self.input_rbf_center_start, self.input_rbf_center_end,
            self.input_rbf_trainable, 
            device=self.device, dtype=self.dtype)

        return

    def __str__(self):
        return self.input_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'input_type': self.input_type,
            'input_n_atombasis': self.input_n_atombasis,
            'input_radial_fn': str(self.input_radial_fn),
            'input_n_radialbasis': self.input_n_radialbasis,
            'input_radial_cutoff': self.input_radial_cutoff,
            'input_cutoff_fn': str(self.input_cutoff_fn),
            'input_rbf_trainable': self.input_rbf_trainable,
            'input_n_maxatom': self.input_n_maxatom,
            }

    def forward(
        self, 
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        pbc_offset: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass of the input module.

        Parameters
        ----------
        atomic_numbers : torch.Tensor(N_atoms)
            Atomic numbers of the system
        positions : torch.Tensor(N_atoms, 3)
            Atomic positions of the system
        idx_i : torch.Tensor(N_pairs)
            Atom i pair index
        idx_j : torch.Tensor(N_pairs)
            Atom j pair index
        pbc_offset : torch.Tensor(N_pairs, 3), optional, default None
            Position offset from periodic boundary condition

        Returns
        -------
        features: torch.tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        distances: torch.tensor(N_pairs)
            Atom pair distances
        vectors: torch.tensor(N_pairs, 3)
            Atom pair vectors
        cutoffs: torch.tensor(N_pairs)
            Atom pair distance cutoffs
        rbfs: torch.tensor(N_pairs, n_radialbasis)
            Atom pair radial basis functions

        """
        
        # Collect atom feature vectors
        features = self.atom_features[atomic_numbers]
        
        # Compute pair connection vector
        if pbc_offset is None:
            vectors = positions[idx_j] - positions[idx_i]
        else:
            vectors = positions[idx_j] - positions[idx_i] + pbc_offset

        # Compute pair distances
        distances = torch.norm(vectors, dim=-1)
        
        # Compute radial basis functions
        rbfs, cutoff = self.input_radial_fn(distances)
        
        return features, distances, vectors, cutoff, rbfs


#======================================
# Graph Module
#======================================

class Graph_PaiNN(torch.nn.Module): 
    """
    PaiNN message passing module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    graph_stability_constant: float, optional, default 1.e-8
        Numerical stability constant added to scalar products of Cartesian 
        information vectors (guaranteed to be non-zero).

    """
    
    # Default arguments for graph module
    _default_args = {
        'graph_n_blocks':               5,
        'graph_activation_fn':          'shifted_softplus',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'graph_n_blocks':               [utils.is_integer],
        'graph_activation_fn':          [utils.is_string, utils.is_callable],
        }

    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        graph_stability_constant: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize NNP graph model.

        """
        
        super(Graph_PaiNN, self).__init__()
        graph_type = 'PaiNN'
        
        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.dtype = config.get('dtype')
        self.device = config.get('device')
        
        # Get input to graph module interface parameters 
        self.n_atombasis = config.get('input_n_atombasis')
        self.n_radialbasis = config.get('input_n_radialbasis')
        
        ####################################
        # # # Graph Module Class Setup # # #
        ####################################
        
        # Initialize activation function
        self.activation_fn = layer.get_activation_fn(
            self.graph_activation_fn)
        
        # Initialize feature-wise, continuous-filter convolution network
        self.descriptors_filter = layer.DenseLayer(
            self.n_radialbasis,
            self.graph_n_blocks*self.n_atombasis*3,
            activation_fn=None,
            bias=True,
            device=self.device,
            dtype=self.dtype)
        
        # Initialize message passing blocks
        self.interaction_block = torch.nn.ModuleList([
            painn_layers.PaiNNInteraction(
                self.n_atombasis, 
                activation_fn=self.activation_fn,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])
        self.mixing_block = torch.nn.ModuleList([
            painn_layers.PaiNNMixing(
                self.n_atombasis, 
                activation_fn=self.activation_fn,
                stability_constant=self.graph_stability_constant,
                device=self.device,
                dtype=self.dtype)
            for _ in range(self.graph_n_blocks)
            ])

        return

    def __str__(self):
        return self.graph_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'graph_type': self.graph_type,
            'graph_n_blocks': self.graph_n_blocks,
            'graph_activation_fn': str(self.graph_activation_fn),
            'graph_stability_constant': self.graph_stability_constant,
            }

    def forward(
        self, 
        features: torch.Tensor,
        distances: torch.Tensor,
        vectors: torch.Tensor,
        cutoff: torch.Tensor,
        rbfs: torch.Tensor,
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> List[torch.Tensor]:

        """
        Forward pass of the graph model.
        
        Parameter
        ---------
        features: torch.tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        distances: torch.tensor(N_pairs)
            Atom pair distances
        vectors : torch.Tensor
            Atom pair connection vectors
        cutoffs: torch.tensor(N_pairs)
            Atom pair distance cutoffs
        rbfs: torch.tensor(N_pairs, n_radialbasis)
            Atom pair radial basis functions
        idx_i : torch.Tensor(N_pairs)
            Atom i pair index
        idx_j : torch.Tensor(N_pairs)
            Atom j pair index

        Returns
        -------
        features_list: [torch.tensor(N_atoms, n_atombasis)]*1
            List of modified atomic feature vectors

        """
        
        # Apply feature-wise, continuous-filter convolution
        descriptors = (
            self.descriptors_filter(rbfs[:, None, :])*cutoff[:, None, None])
        descriptors_list = torch.split(
            descriptors, 3*self.n_atombasis, dim=-1)
        
        # Normalize atom pair vectors
        vectors_normalized = vectors/distances[:, None]

        # Assign isolated atomic feature vectors as scalar feature 
        # vectors
        fsize = features.shape # (len(atomic_numbers), n_atombasis)
        sfeatures = features[:, None, :]

        # Initialize vectorial feature vectors
        vfeatures = torch.zeros((fsize[0], 3, fsize[1]), device=self.device)

        # Apply message passing model to modify from isolated atomic features
        # vectors to molecular atomic features as a function of the chemical
        # environment
        for ii, (interaction, mixing) in enumerate(
            zip(self.interaction_block, self.mixing_block)
        ):
            sfeatures, vfeatures = interaction(
                sfeatures, 
                vfeatures, 
                descriptors_list[ii], 
                vectors_normalized, 
                idx_i, 
                idx_j,
                fsize[0], 
                fsize[1])
            sfeatures, vfeatures = mixing(
                sfeatures, 
                vfeatures, 
                fsize[1])
        
        # Flatten scalar atomic feature vector
        sfeatures = sfeatures.squeeze(1)
        
        return sfeatures, vfeatures


#======================================
# Output Module
#======================================

class Output_PaiNN(torch.nn.Module): 
    """
    PaiNN output module class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    output_properties: list(str), optional '['energy', 'forces']'
        List of output properties to compute by the model
        e.g. ['energy', 'forces', 'atomic_charges']
    output_n_residual: int, optional, default 1
        Number of residual layers for transformation from atomic feature vector
        to output results.
    output_activation_fn: (str, callable), optional, default 'shifted_softplus'
        Residual layer activation function.
    output_block_options: dict(str, Any), optional, default None
        ...
    output_scaling_parameter: dictionary, optional, default None
        Property average and standard deviation for the use as scaling factor 
        (standard deviation) and shift term (average) parameter pairs
        for each property.
    **kwargs: dict, optional
        Additional arguments

    """
    
    # Default arguments for graph module
    _default_args = {
        'output_properties':            None,
        'output_n_residual':            1,
        'output_activation_fn':         'shifted_softplus',
        'output_block_options':         {},
        'output_scaling_parameter':     None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'output_properties':            [utils.is_string_array, utils.is_None],
        'output_n_residual':            [utils.is_integer],
        'output_activation_fn':         [utils.is_string, utils.is_callable],
        'output_block_options':         [utils.is_dictionary],
        'output_scaling_parameter':     [utils.is_dictionary],
        }
    
    # Property exclusion lists for properties (keys) derived from other 
    # properties (items) but in the model class
    _property_exclusion = {
        'energy': ['atomic_energies'],
        'forces': [], 
        'hessian': [],
        'dipole': ['atomic_charges', 'atomic_dipole']}
        
    # Output module specially handled properties
    #_property_special = ['energy', 'atomic_energies', 'atomic_charges']
    
    # Default output block options for atom-wise scalar properties such as, 
    # e.g., 'atomic_energies'.
    _default_output_scalar = {
        'n_property':           1,
        'n_layer':              2,
        'n_neurons':            None,
        'activation_fn':        torch.nn.functional.silu,
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.zeros_,
        'weight_init_last':     torch.nn.init.zeros_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    # Default output block options for atom-wise vector properties such as, 
    # e.g., 'atomic_dipole'.
    _default_output_vector = {
        'n_property':           1,
        'n_layer':              2,
        'n_neurons':            None,
        'activation_fn':        torch.nn.functional.silu,
        'bias_layer':           True,
        'bias_last':            True,
        'weight_init_layer':    torch.nn.init.zeros_,
        'weight_init_last':     torch.nn.init.zeros_,
        'bias_init_layer':      torch.nn.init.zeros_,
        'bias_init_last':       torch.nn.init.zeros_,
        }
    
    
    # Default output block assignment to properties
    # key: output property
    # item: list -> [0]:    str, Output block type 'scalar' or 'vector'
    #                       None, no output block (skip property)
    #               [1]:    dict, output block options if [0] str
    #                       str, dependency to other property if [0] None
    #               [2]:    optional, int: scalar or vector result from a
    #                           'vector' type output block
    #       dict -> key:    str, either default case or case if respective
    #                           property included in output properties list
    #               item:   list -> [0]: Output block type (see item -> list)
    #                               [1]: Output block options ...
    #                               [2]: 'Vector' output block result ...
    _default_property_assignment = {
        'energy': [None, 'atomic_energies'],
        'atomic_energies': ['scalar', _default_output_scalar],
        'forces': [None, 'energy'],
        'dipole': [None, 'atomic_charges'],
        'atomic_charges': {
            'default': ['scalar', _default_output_scalar]
            'atomic_dipoles': ['vector', _default_output_vector, 0]
            },
        'atomic_dipoles': ['vector', _default_output_vector, 1],
        }
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        output_properties: Optional[List[str]] = None,
        output_n_residual: Optional[int] = None,
        output_activation_fn: Optional[Union[str, object]] = None,
        output_block_options: Optional[Dict[str, Any]] = None,
        output_scaling_parameter: Optional[Dict[str, List[float]]] = None,
        **kwargs
    ):
        """
        Initialize PaiNN output model.

        """

        super(Output_PaiNN, self).__init__()
        output_type = 'PaiNN'

        ####################################
        # # # Check Module Class Input # # #
        ####################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.dtype = config.get('dtype')
        self.device = config.get('device')

        # Get input and graph to output module interface parameters 
        self.n_maxatom = config.get('input_n_maxatom')
        self.n_atombasis = config.get('input_n_atombasis')
        self.n_blocks = config.get('graph_n_blocks')

        ##########################################
        # # # Check Output Module Properties # # #
        ##########################################

        # Get model properties to check with output module properties
        model_properties = config.get('model_properties')

        # Initialize output module properties
        properties_list = []
        properties_assignment = {}
        
        # Check defined output properties
        if self.output_properties is not None:
            for prop in self.output_properties:
                # Check if property is available
                if prop in self._default_property_assignment:
                    instructions = self._default_property_assignment[prop]
                    # If skip label is found in instructions
                    if utils.is_list(instructions) and instructions[0] is None:
                        # Check dependencies and skip property
                        if not (
                            instructions[1] in self.output_properties
                            or instructions[1] in model_properties
                        ):
                            raise SyntaxError(
                                "Model property prediction requires property "
                                + f"{instructions[1]:s}, which is not "
                                + "defined!")
                        else:
                            pass
                    elif utils.is_list(instructions):
                        properties_list.append(prop)
                        if output_block_options
                        properties_assignment.append(instructions)
                else:
                    raise NotImplementedError(
                        f"Output module of type {self.output_type:s} "
                        + "does not support the property prediction of "
                        + f"{prop:s}!")

        # Check output module properties with model properties
        for prop in model_properties:
            if (
                prop not in properties_list
                and prop in self._property_exclusion
            ):
                for prop_der in self._property_exclusion[prop]:
                    if prop_der not in properties_list:
                        properties_list.append(prop_der)
            elif prop not in properties_list:
                properties_list.append(prop)

        # Update output property list and global configuration dictionary
        self.output_properties = properties_list
        config_update = {
            'output_properties': self.output_properties}
        config.update(config_update)

        #####################################
        # # # Output Module Class Setup # # #
        #####################################
        
        
        

        ## Collect output block properties
        #self.output_block_options = settings._default_output_block_properties
        #self.output_block_options.update(output_block_options)

        ## Initialize output block dictionary for properties
        #self.output_block_dict = torch.nn.ModuleDict({})
        
        ## Create output layer for properties with certain exceptions
        #for prop in self.output_properties:
            
            ## No output_block for derivatives of properties such as atom 
            ## forces, Hessian or molecular dipole
            #if prop in ['forces', 'hessian', 'dipole']:
                #continue
            
            ## Initialize output block
            #self.output_block_dict[prop] = self.create_output_block(
                #self.input_n_atombasis,
                #self.output_block_options['n_outputneurons'],
                #self.output_block_options['n_hiddenlayers'],
                #n_hiddenneurons=self.output_block_options['n_hiddenneurons'],
                #activation_fn=self.activation_fn,
                #output_bias=self.output_block_options['output_bias'],
                #output_init_zero=self.output_block_options['output_init_zero'],
                #device=self.device,
                #dtype=self.dtype,
                #)
            
            
