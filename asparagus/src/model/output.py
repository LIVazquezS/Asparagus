
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import torch
#import pytorch_lightning as pl

from .. import settings
from .. import utils
from .. import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_output_model', 'Output_PhysNet']

#======================================
# Output Models
#======================================

class Output_PhysNet(torch.nn.Module): 
    """
    PhysNet Output model


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    output_n_residual: int, optional, default 1
        Number of residual layers for message refinement
    output_properties: list(str), optional '['energy', 'forces']'
        List of output properties to compute by the model
        e.g. ['energy', 'forces', 'atomic_charges']
    output_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    **kwargs: dict, optional
        Additional arguments

    Returns
    -------
    callable object
        PhysNet Output model object

    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        output_n_residual: Optional[int] = None,
        output_properties: Optional[List[str]] = None,
        output_activation_fn: Optional[Union[str, object]] = None,
        **kwargs
    ):
        """
        Initialize PhysNet output model.

        """

        super().__init__()

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
        self.output_type = 'PhysNetOut'

        # Initialize activation function
        self.output_activation_fn = layers.get_activation_fn(
            self.output_activation_fn)

        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        # Get output model interface parameters 
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.graph_n_blocks = config.get('graph_n_blocks')
        self.model_properties = config.get('model_properties')

        # Assign graph model training parameters
        self.rate = settings._global_rate

        # Update 'output_properties' with 'model_properties'
        for prop in self.model_properties:
            if prop not in self.output_properties:
                self.output_properties.append(prop)

        # Initialize property to output block dictionary
        self.output_property_block = torch.nn.ModuleDict({})

        # Initialize property to number of output block predictions dictionary
        self.output_property_num = {}

        # Check special case: atom energies and charges from one output block
        if all([
            prop in self.output_properties
            for prop in ['energy', 'atomic_charges']]
        ):

            # Set case flag for output module predicting atomic energies and
            # charges
            self.output_energies_charges = True

            # PhysNet energy and atom charges output block
            output_block = torch.nn.ModuleList([
                layers.OutputBlock(
                    self.input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=2,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(self.graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block['atomic_energies_charges'] = (
                output_block)
            self.output_property_num['atomic_energies_charges'] = 2

        elif 'energy' in self.output_properties:

            # Set case flag for output module predicting just atomic energies
            # or charges
            self.output_energies_charges = False

            # PhysNet energy only output block
            output_block = torch.nn.ModuleList([
                layers.OutputBlock(
                    self.input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=1,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(self.graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block['atomic_energies'] = output_block
            self.output_property_num['atomic_energies'] = 1

        elif 'atomic_charges' in self.output_properties:

            # Set case flag for output module predicting just atomic energies
            # or charges
            self.output_energies_charges = False

            # PhysNet atomic_charges only output block (A bit meaningless tbh)
            output_block = torch.nn.ModuleList([
                layers.OutputBlock(
                    self.input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=1,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(self.graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block['atomic_charges'] = output_block
            self.output_property_num['atomic_charges'] = 1

        else:

            # Set case flag for output module predicting just atomic energies
            # or charges
            self.output_energies_charges = False

        # Initialize property scaling factor and shifting term
        self.output_scaling = {}

        # Create further output blocks for properties with certain exceptions
        for prop in self.output_properties:

            # No output_block for already covered atomic energies and 
            # atomic charges as well as derivatives such as atom forces, 
            # Hessian or molecular dipole
            if prop in [
                    'energy', 'atomic_charges', 'forces', 'hessian', 'dipole']:
                continue

            # Initialize output block
            output_block = torch.nn.ModuleList([
                layers.OutputBlock(
                    self.input_n_atombasis,
                    self.output_n_residual,
                    self.output_activation_fn,
                    output_n_results=1,
                    rate=self.rate,
                    device=self.device,
                    dtype=self.dtype)
                for _ in range(self.graph_n_blocks)
                ])

            # Assign output block to dictionary
            self.output_property_block[prop] = output_block
            self.output_property_num[prop] = 1


    def forward(
        self,
        messages_list: List[torch.Tensor],
        properties: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of PhysNet output model

        Parameters
        ----------
        messages_list : list(torch.Tensor)
            List of messages from PhysNet graph model
        properties : list(str), optional
            List of properties to compute by the model
            e.g. ['energy', 'forces', 'atomic_charges']

        Returns
        -------
        dict(str, torch.Tensor)
            Dictionary of predicted properties

        """

        # Initialize predicted properties dictionary
        output_prediction = {}
        
        # Iterate over output blocks
        for prop, output_block in self.output_property_block.items():

            # Skip if property not requested
            if properties is not None and prop not in properties:
                continue

            # Compute prediction and loss function contribution
            nhloss = 0.0
            last_prediction2 = 0.0
            for iblock, (message, output_layer) in enumerate(
                    zip(messages_list, output_block)):
                prediction = output_layer(message)
                if iblock:
                    output_prediction[prop] = (
                        output_prediction[prop] + prediction)
                else:
                    output_prediction[prop] = prediction
                
                # If training mode is active, compute nhloss contribution
                if self.training:
                    prediction2 = prediction**2
                    if iblock:
                        nhloss = nhloss + torch.mean(
                            prediction2
                            / (prediction2 + last_prediction2 + 1.0e-7)
                            )
                    last_prediction2 = prediction2
            
            # Save nhloss if training mode is active
            if self.training:
                output_prediction['nhloss'] = nhloss
            
            # Flatten prediction for scalar properties
            if self.output_property_num[prop] == 1:
                output_prediction[prop] = torch.flatten(
                    output_prediction[prop], start_dim=0)
            
        # Post-process atomic energies/charges case
        if self.output_energies_charges:

            output_prediction['atomic_energies'], \
                output_prediction['atomic_charges'] = (
                    output_prediction['atomic_energies_charges'][:, 0],
                    output_prediction['atomic_energies_charges'][:, 1])

        return output_prediction


    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """

        return {
            'output_n_residual': self.output_n_residual,
            'output_properties': self.output_properties,
            }


class Output_PaiNN(torch.nn.Module): 
    """
    PaiNN Output model


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    output_n_residual: int, optional, default 1
        Number of residual layers for message refinement
    output_properties: list(str), optional '['energy', 'forces']'
        List of output properties to compute by the model
        e.g. ['energy', 'forces', 'atomic_charges']
    output_activation_fn: (str, object), optional, default 'shifted_softplus'
        Activation function
    **kwargs: dict, optional
        Additional arguments

    Returns
    -------
    callable object
        PaiNN Output model object

    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        output_n_residual: Optional[int] = None,
        output_properties: Optional[List[str]] = None,
        output_block_options: Optional[Dict[str, Any]] = None,
        output_activation_fn: Optional[Union[str, object]] = None,
        **kwargs
    ):
        """
        Initialize PaiNN output model.

        """
        
        super().__init__()

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
        self.output_type = 'PaiNNOut'

        # Initialize activation function
        self.output_activation_fn = layers.get_activation_fn(
            self.output_activation_fn)

        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        # Get output model interface parameters 
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.model_properties = config.get('model_properties')
        
        # Assign graph model training parameters
        self.rate = settings._global_rate

        # Update 'output_properties' with 'model_properties'
        for prop in self.model_properties:
            if prop not in self.output_properties:
                self.output_properties.append(prop)

        # Collect output block properties
        self.output_block_options = settings._default_output_block_properties
        self.output_block_options.update(output_block_options)

        # Initialize output block dictionary for properties
        self.output_block_dict = torch.nn.ModuleDict({})
        
        # Create output layer for properties with certain exceptions
        for prop in self.output_properties:
            
            # No output_block for derivatives of properties such as atom 
            # forces, Hessian or molecular dipole
            if prop in ['forces', 'hessian', 'dipole']:
                continue
            
            # Initialize output block
            self.output_block_dict[prop] = self.create_output_block(
                self.input_n_atombasis,
                self.output_block_options['n_outputneurons'],
                self.output_block_options['n_hiddenlayers'],
                n_hiddenneurons=self.output_block_options['n_hiddenneurons'],
                activation_fn=self.activation_fn,
                output_bias=self.output_block_options['output_bias'],
                output_init_zero=self.output_block_options['output_init_zero'],
                device=self.device,
                dtype=self.dtype,
                )
            
            
    def create_output_block(
        self,
        n_inputneurons: int,
        n_outputneurons: int,
        n_hiddenlayers: int,
        n_hiddenneurons: Optional[Union[int, List[int]]] = None,
        activation_fn: Optional[object] = None,
        output_bias: Optional[bool] = True,
        output_init_zero: Optional[bool] = False,
        device: Optional[str] = 'cpu',
        dtype: Optional[object] = torch.float64,
    ):
        
        # Check hidden layer neuron option
        if n_hiddenlayers:
            if n_hiddenneurons is None:
                # Half number of hidden layer neurons with each layer
                n_neurons = n_inputneurons
                n_hiddenneurons = [n_inputneurons]
                for ii in range(n_hiddenlayers):
                    n_neurons = max(n_outputneurons, n_neurons//2)
                    n_hiddenneurons.append(n_neurons)
            elif utils.is_integer(n_hiddenneurons):
                n_hiddenneurons = (
                    [n_inputneurons] + [n_hiddenneurons]*(n_hiddenlayers))
        else:
            # If no hidden layer, set hidden neurons to property neuron number
            n_hiddenneurons = [n_inputneurons]

        # Initialize output module
        output_block = torch.nn.Sequential(
            layers.DenseLayer(
                Nin=n_inputneurons, 
                Nout=n_hiddenneurons[0],
                activation_fn=activation_fn,
                device=device,
                dtype=dtype
                ),
            )
        
        # Append hidden layers
        for ii in range(n_hiddenlayers):
            output_block.append(
                layers.DenseLayer(
                    Nin=n_hiddenneurons[ii], 
                    Nout=n_hiddenneurons[ii + 1],
                    activation_fn=activation_fn,
                    device=device,
                    dtype=dtype
                    ),
                )
        
        # Append output layer
        output_block.append(
            layers.DenseLayer(
                Nin=n_hiddenneurons[-1], 
                Nout=n_outputneurons,
                activation_fn=None,
                bias=output_bias,
                W_init=output_init_zero,
                device=device,
                dtype=dtype
                ),
            )

        return output_block

#======================================
# Output Model Assignment  
#======================================

output_model_available = {
    'PhysNetOut'.lower(): Output_PhysNet,
    'PaiNNOut'.lower(): Output_PaiNN,
    }

def get_output_model(
    config: Optional[Union[str, dict, object]] = None,
    output_type: Optional[str] = None,
    **kwargs
):
    """
    Output module selection

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    output_type: str
        Output model transforming features into demanded properties
    **kwargs: dict, optional
        Additional arguments for parameter initialization 

    Returns
    -------
    callable object
        Output model object to transform features into demanded properties
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

    # Check output model type
    if config.get('output_type') is None:
        model_type = config.get('model_type')
        if settings._available_output_model.get(model_type) is None:
            raise SyntaxError(
                "No output model type could assigned from defined model "
                + f"type '{model_type:s}'!")
        config['output_type'] = settings._available_output_model.get(
            model_type)
    output_type = config['output_type']

    # Output model type assignment
    if (
        output_type.lower() in 
        [key.lower() for key in output_model_available.keys()]
    ):
        return output_model_available[output_type.lower()](
            config,
            **kwargs)
    else:
        raise ValueError(
            f"Output model type output '{output_type:s}' is not valid!\n" +
            "Choose from:\n" + str(output_model_available.keys()))
    
    return

