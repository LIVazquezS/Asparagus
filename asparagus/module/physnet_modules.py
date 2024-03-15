import logging
from typing import Optional, Union, List, Callable, Any

import torch

from .. import layer
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Input_PhysNet', 'Graph_PhysNet', 'Output_PhysNet']

#======================================
# Input Module
#======================================

class Input_PhysNet(torch.nn.Module):
    """
    PhysNet input module class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    input_n_atombasis: int, optional, default 128
        Number of atomic features (length of the atomic feature vector)
    input_n_radialbasis: int, optional, default 64
        Number of input radial basis centers
    input_radial_cutoff: float, optional, default 8.0
        Cutoff distance radial basis function
    input_cutoff_fn: (str, callable), optional, default 'Poly6'
        Cutoff function type for radial basis function scaling
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
    input_atom_features_range: float, optional, default sqrt(3)
        Range for uniform distribution randomizer for initial atom feature
        vector.
    **kwargs: dict, optional
        Additional arguments for parameter initialization

    """
    
    # Default arguments for input module
    _default_args = {
        'input_type':                   None,
        'input_n_atombasis':            128,
        'input_n_radialbasis':          64,
        'input_radial_cutoff':          8.0,
        'input_cutoff_fn':              'Poly6',
        'input_rbf_center_start':       1.0,
        'input_rbf_center_end':         None,
        'input_rbf_trainable':          True,
        'input_n_maxatom':              94,
        'input_atom_features_range':    np.sqrt(3),
        }

    # Expected data types of input variables
    _dtypes_args = {
        'input_type':                   [utils.is_string],
        'input_n_atombasis':            [utils.is_integer],
        'input_n_radialbasis':          [utils.is_integer],
        'input_radial_cutoff':          [utils.is_numeric],
        'input_cutoff_fn':              [utils.is_string, utils.is_callable],
        'input_rbf_center_start':       [utils.is_numeric],
        'input_rbf_center_end':         [utils.is_None, utils.is_numeric],
        'input_rbf_trainable':          [utils.is_bool],
        'input_n_maxatom':              [utils.is_integer],
        'input_atom_features_range':    [utils.is_numeric],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        input_type: Optional[str] = None,
        input_n_atombasis: Optional[int] = None,
        input_n_radialbasis: Optional[int] = None,
        input_radial_cutoff: Optional[float] = None,
        input_cutoff_fn: Optional[Union[str, object]] = None,
        input_rbf_center_start: Optional[float] = None,
        input_rbf_center_end: Optional[float] = None,
        input_rbf_trainable: Optional[bool] = None,
        input_n_maxatom: Optional[int] = None,
        input_atom_features_range: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize PhysNet input module.

        """

        super(Input_PhysNet, self).__init__()
        
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
            argitems=locals().items(),
            argsskip=['self', 'config', 'metadata', 'kwargs', '__class__'],
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )
            
        # Update global configuration dictionary
        config.update(config_update)
        
        # Assign module variable parameters from configuration
        self.dtype = config.get('dtype')
        self.device = config.get('device')
        
        # Check general model cutoff with radial basis cutoff
        if config.get('model_cutoff') is None:
            config['model_cutoff'] = self.input_radial_cutoff
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
        self.atom_features = nn.Embedding(
            self.input_n_maxatom + 1,
            self.input_n_atombasis, 
            padding_idx=0)
        #self.atom_features = torch.nn.Parameter(
            #torch.empty(
                #self.input_n_maxatom + 1, self.input_n_atombasis,
                #dtype=self.dtype,
                #device=self.device
                #).uniform_(
                    #-self.input_atom_features_range, 
                    #self.input_atom_features_range)
            #)
        
        # Initialize radial cutoff function
        self.input_cutoff_fn = layer.get_cutoff_fn(self.input_cutoff_fn)(
            self.input_radial_cutoff)
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_radial_cutoff
        
        # Initialize Radial basis function
        radial_fn = layer.get_radial_fn(self.input_type)
        self.input_radial_fn = radial_fn(
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
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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
        cutoffs: torch.tensor(N_pairs)
            Atom pair distance cutoffs
        rbfs: torch.tensor(N_pairs, n_radialbasis)
            Atom pair radial basis functions

        """
        
        # Collect atom feature vectors
        features = self.atom_features[atomic_numbers]

        # Compute atom pair distances
        if pbc_offset is None:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i],
                dim=-1)
        else:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i] + pbc_offset,
                dim=-1)

        # Compute distance cutoff values
        cutoffs = self.input_cutoff_fn(distances)

        # Compute radial basis functions
        rbfs = self.input_radial_fn(distances)

        return features, distances, cutoffs, rbfs


#======================================
# Graph Module
#======================================

class Graph_PhysNet(torch.nn.Module): 
    """
    PhysNet message passing module class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    graph_n_blocks: int, optional, default 5
        Number of information processing cycles
    graph_n_residual_interaction: int, optional, default 3
        Number of residual layers for atomic feature and radial basis vector
        interaction.
    graph_n_residual_features: int, optional, default 2
        Number of residual layers for atomic feature interactions.
    graph_activation_fn: (str, object), optional, default 'shifted_softplus'
        Residual layer activation function.
    **kwargs: dict, optional
        Additional arguments

    """
    
    # Default arguments for graph module
    _default_args = {
        'graph_type':                   'PhysNet',
        'graph_n_blocks':               5,
        'graph_n_residual_interaction': 3,
        'graph_n_residual_features':    2,
        'graph_activation_fn':          'Poly6',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'graph_type':                   [utils.is_string, utils.is_None],
        'graph_n_blocks':               [utils.is_integer],
        'graph_n_residual_interaction': [utils.is_integer],
        'graph_n_residual_features':    [utils.is_integer],
        'graph_activation_fn':          [utils.is_string, utils.is_callable],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        graph_type: Optional[str] = None,
        graph_n_blocks: Optional[int] = None,
        graph_n_residual_interaction: Optional[int] = None,
        graph_n_residual_features: Optional[int] = None,
        graph_activation_fn: Optional[Union[str, object]] = None,
        **kwargs
    ):
        """
        Initialize PhysNet message passing module.

        """
        
        super(Graph_PhysNet).__init__()
        
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
            argitems=locals().items(),
            argsskip=['self', 'config', 'metadata', 'kwargs', '__class__'],
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.dtype = config.get('dtype')
        self.device = config.get('device')

        # Get input to graph module interface parameters 
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.input_n_radialbasis = config.get('input_n_radialbasis')
        
        ####################################
        # # # Graph Module Class Setup # # #
        ####################################
        
        # Initialize activation function
        self.graph_activation_fn = layer.get_activation_fn(
            self.graph_activation_fn)

        # Initialize message passing blocks
        self.interaction_blocks = torch.nn.ModuleList([
            layer.physnet_layers.InteractionBlock(
                self.input_n_atombasis, 
                self.input_n_radialbasis, 
                self.graph_n_residual_interaction,
                self.graph_n_residual_features,
                self.graph_activation_fn,
                device=self.device,
                dtype=self.dtype,
                )
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
            'graph_n_residual_interaction': self.graph_n_residual_interaction,
            'graph_n_residual_features': self.graph_n_residual_features,
            }

    def forward(
        self, 
        features: torch.Tensor,
        distances: torch.Tensor,
        cutoffs: torch.Tensor, 
        idx_i: torch.Tensor, 
        idx_j: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward pass of the graph module.
        
        Parameters
        ----------
        features: torch.tensor(N_atoms, n_atombasis)
            Atomic feature vectors
        distances: torch.tensor(N_pairs)
            Atom pair distances
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
        features_list: [torch.tensor(N_atoms, n_atombasis)]*n_blocks
            List of modified atomic feature vectors

        """
        
        # Compute descriptor vectors
        descriptors = cutoffs*rbfs
        
        # Initialize refined feature vector list
        x_list = []
        
        # Apply message passing model
        x = features
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, descriptors, idx_i, idx_j)
            x_list.append(x)

        return x_list


#======================================
# Output Module
#======================================

class Output_PhysNet(torch.nn.Module): 
    """
    PhysNet output module class

    Parameters
    ----------
    config: (str, dict, object)
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
    output_scaling_parameter: dictionary, optional, default None
        Property average and standard deviation for the use as scaling factor 
        (standard deviation) and shift term (average) parameter pairs
        for each property.
    **kwargs: dict, optional
        Additional arguments

    """

    # Default arguments for graph module
    _default_args = {
        'output_type':                  'PhysNet',
        'output_properties':            None,
        'output_n_residual':            1,
        'output_activation_fn':         'Poly6',
        'output_scaling_parameter':     None,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'output_type':                  [utils.is_string],
        'output_properties':            [utils.is_string_array, utils.is_None],
        'output_n_residual':            [utils.is_integer],
        'output_activation_fn':         [utils.is_string, utils.is_callable],
        'output_scaling_parameter':     [utils.is_dictionary],
        }
    
    # Property exclusion lists for properties derived from other properties 
    # or are treated specially.
    _property_derived = ['forces', 'hessian', 'dipole']
    _property_special = ['energy', 'atomic_energies', 'atomic_charges']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        output_type: Optional[List[str]] = None,
        output_n_residual: Optional[int] = None,
        output_activation_fn: Optional[Union[str, object]] = None,
        output_scaling_parameter: Optional[Dict[str, List[float]]] = None,
        **kwargs
    ):
        """
        Initialize PhysNet output module.

        """

        super(Output_PhysNet).__init__()

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
            argitems=locals().items(),
            argsskip=['self', 'config', 'metadata', 'kwargs', '__class__'],
            check_default=utils.get_default_args(self, module),
            check_dtype=utils.get_dtype_args(self, module)
        )

        # Update global configuration dictionary
        config.update(config_update)

        # Assign module variable parameters from configuration
        self.dtype = config.get('dtype')
        self.device = config.get('device')

        # Get input and graph to output module interface parameters 
        self.input_n_maxatom = config.get('input_n_maxatom')
        input_n_atombasis = config.get('input_n_atombasis')
        graph_n_blocks = config.get('graph_n_blocks')

        # Get model properties to check with output module properties
        model_properties = config.get('model_properties')

        ##########################################
        # # # Check Output Module Properties # # #
        ##########################################

        # Check output module properties
        if self.output_properties is None:
            self.output_properties = []
        for model_prop in model_properties:
            if model_prop not in self.output_properties:
                self.output_properties.append(model_prop)

        #####################################
        # # # Output Module Class Setup # # #
        #####################################

        # Initialize activation function
        self.output_activation_fn = layer.get_activation_fn(
            self.output_activation_fn)

        # Initialize property to output block dictionary
        self.output_property_block = torch.nn.ModuleDict({})

        # Initialize property to number of output block predictions dictionary
        self.output_n_property = {}

        # Check special case: atom energies and charges from one output block
        if all([
            prop in self.output_properties
            for prop in ['atomic_energies', 'atomic_charges']]
        ):

            # Set case flag for output module predicting atomic energies and
            # charges
            self.output_energies_charges = True

            # PhysNet energy and atom charges output block
            output_block = torch.nn.ModuleList([
                layer.physnet_layers.OutputBlock(
                    input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=2,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(graph_n_blocks)]
                )

            # Assign output block to dictionary
            self.output_property_block['atomic_energies_charges'] = (
                output_block)
            self.output_n_property['atomic_energies_charges'] = 2

        if any([
            prop in self.output_properties
            for prop in ['atomic_energies', 'atomic_charges']]
        ):

            # Set case flag for output module predicting just atomic energies
            # or charges
            self.output_energies_charges = False

            # PhysNet energy only output block
            output_block = torch.nn.ModuleList([
                layer.physnet_layers.OutputBlock(
                    input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=1,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block[prop] = output_block
            self.output_property_num[prop] = 1

        else:

            # Set case flag for output module predicting just atomic energies
            # or charges
            self.output_energies_charges = False

        # Initialize property scaling dictionary and atomic energy shift
        self.set_property_scaling(self.output_scaling_parameter)
        
        # Create further output blocks for properties with certain exceptions
        for prop in self.output_properties:

            # Skip deriving properties
            if prop in self._property_derived:
                continue

            # Initialize scaling factor and shifting term if not done already
            if prop not in self.output_scaling:
                self.output_scaling[prop] = torch.nn.Parameters(
                    torch.Tensor(
                        [[1.0, 0.0] for _ in range(self.input_n_maxatom)],
                        device=self.device, 
                        dtype=self.dtype)
                    )

            # No output_block for already covered special properties
            if prop in self._property_special:
                continue

            # Initialize output block
            output_block = torch.nn.ModuleList([
                layer.physnet_layers.OutputBlock(
                    input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=1,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block[prop] = output_block
            self.output_property_num[prop] = 1

        return

    def __str__(self):
        return self.output_type
    
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {
            'output_type': self.output_type,
            'output_properties': self.output_properties,
            'output_unit_properties': self.output_unit_properties,
            'output_n_residual': self.output_n_residual,
            'output_activation_fn': self.output_activation_fn
            }

    def set_property_scaling(
        self,
        scaling_parameter: Dict[str, List[float]],
    ):
        """
        Update output property scaling factor and shift term dictionary.
        
        Parameters
        ----------
        scaling_parameter: dict(str, torch.Tensor(2))
            Property average and standard deviation for the use as
            scaling factor (standard deviation) and shift term (average) 
            parameter pairs (item) for each property (key).

        """

        # Initialize property scaling dictionary
        if not hasattr(self, 'output_scaling'):
            self.output_scaling = {}
        
        # Check property scaling input
        if scaling_parameter is None:
            return

        # Assign scaling factor and shift term
        for prop, (shift, scale) in scaling_parameter.items():
            
            # Skip for deriving properties and special properties
            if prop in self._property_derived:
                continue

            self.output_scaling[prop] = torch.nn.Parameters(
                torch.Tensor(
                    [[scale, shift] for _ in range(self.input_n_maxatom)],
                    device=self.device, 
                    dtype=self.dtype)
                )
        
        return

    def set_atomic_energies_shift(
        self,
        atomic_energies_shifts: Dict[Union[int, str], float],
    ):
        """
        Set atom type related atomic energies shift terms
        
        Parameters
        ----------
        atomic_energies_shifts: dict(str, torch.Tensor(1))
            Atom type related atomic energies shift for core electron energy
            contribution.

        """
        
        # Check atomic energies shift input
        if atomic_energies_shifts is None:
            return
        
        # Set atomic energies
        for atom_type, shift in atomic_energies_shifts.items():
            
            # Check atom type and energy shift definition
            if utils.is_string(atom_type):
                atomic_number = utils.data.atomic_numbers.get(
                    atom_type.lower())
                if atomic_number is None:
                    raise SyntaxError(
                        f"Atom type symbol '{atom_type:s}' is not known "
                        + "(comparison is not case sensetive)!")
            elif utils.is_numeric(atom_type):
                atomic_number = int(atom_type)
            else:
                raise SyntaxError(
                    f"Atom type symbol data type '{type(atom_type):s}' "
                    + "is not valid! Define either as atomic number (int) or "
                    + "atom symbol (str).")
            
            if not utils.is_numeric(shift):
                raise SyntaxError(
                    f"Atom type energy shift type '{type(shift):s}' "
                    + "is not valid and must be numeric!")
            
            # Set atomic energy shift
            self.output_scaling['atomic_energies'][atomic_number][1] = shift
        
        return

    def forward(
        self,
        features_list: List[torch.Tensor],
        atomic_numbers: Optional[torch.Tensor] = None,
        properties: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of output module

        Parameters
        ----------
        features_list : [torch.Tensor(N_atoms, n_atombasis)]*n_blocks
            List of atom feature vectors
        atomic_numbers: torch.Tensor(N_atoms), optional, default None
            List of atomic numbers
        properties: list(str), optional, default None
            List of properties to compute by the model. If None, all properties
            are predicted.

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary of predicted properties

        """

        # Initialize predicted properties dictionary
        output_prediction = {}
        
        # Check requested properties
        if properties is None:
            predict_all = True
        else:
            predict_all = False

        # Initialize training properties
        if self.training:
            nhloss = 0.0
            last_prediction_squared = 0.0
        
        # Iterate over output blocks
        for prop, output_block in self.output_property_block.items():

            # Skip if property not requested
            if not predict_all and prop not in properties:
                continue

            # Compute prediction and loss function contribution
            for iblock, (features, output) in enumerate(
                zip(messages_list, output_block)
            ):

                prediction = output(features)
                if iblock:
                    output_prediction[prop] = (
                        output_prediction[prop] + prediction)
                else:
                    output_prediction[prop] = prediction
                
                # If training mode is active, compute nhloss contribution
                if self.training:
                    prediction_squared = prediction**2
                    if iblock:
                        nhloss = nhloss + torch.mean(
                            prediction_squared
                            / (
                                prediction_squared 
                                + last_prediction_squared 
                                + 1.0e-7)
                            )
                    last_prediction_squared = prediction_squared
            
            # Flatten prediction for scalar properties
            if self.output_property_num[prop] == 1:
                output_prediction[prop] = torch.flatten(
                    output_prediction[prop], start_dim=0)
            
        # Save nhloss if training mode is active
        if self.training:
            output_prediction['nhloss'] = nhloss
        
        # Post-process atomic energies/charges case
        if self.output_energies_charges:

            output_prediction['atomic_energies'], \
                output_prediction['atomic_charges'] = (
                    output_prediction['atomic_energies_charges'][:, 0],
                    output_prediction['atomic_energies_charges'][:, 1])

        # Apply property scaling
        for prop, scaling in self.output_scaling.items():
            (scale, shift) = scaling[atomic_numbers].T
            output_prediction[prop] = (
                output_prediction[prop]*scale + shift)

        return output_prediction

        
