
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import torch

from .. import model
from .. import layers
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Calculator_PaiNN']

# ======================================
# Calculator Models
# ======================================


class Calculator_PaiNN(torch.nn.Module):
    """
    PaiNN Calculator model


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    model_properties: list(str), optional, default '['energy', 'forces']'
        Properties to predict by calculator model
    model_unit_properties: dict, optional, default None
        Property units of the model prediction. If None, units from the
        reference dataset are taken.
    model_interaction_cutoff: float, optional, default 12.0
        Max. atom interaction cutoff
    **kwargs: dict, optional
        Additional arguments

    Returns
    -------
    callable object
        PaiNN Calculator object for training


    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_descriptor_cutoff: Optional[float] = None,
        model_interaction_cutoff: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize NNP Calculator model.

        """
        
        super(Calculator_PaiNN, self).__init__()
        
        ########################################
        # # # Check PaiNN Calculator Input # # #
        ########################################

        # Get configuration object
        self.config = settings.get_config(config)

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
        self.config.update(config_update)

        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        ###############################
        # # # Prepare PaiNN Input # # #
        ###############################

        # Calculator class type
        self.model_type = 'PaiNN'
        
        # Convert 'model_properties' to list
        self.model_properties = list(self.model_properties)

        # Check for energy gradient properties in prediction list
        if any([
                prop in self.model_properties
                for prop in ['forces', 'hessian']]):
            self.model_gradient = True
            if 'forces' in self.model_properties:
                self.model_forces = True
            else:
                self.model_forces = False
            if 'hessian' in self.model_properties:
                self.model_hessian = True
            else:
                self.model_hessian = False
        else:
            self.model_gradient = False
            self.model_forces = False
            self.model_hessian = False

        #################################
        # # # Prepare PaiNN Modules # # #
        #################################

        # Check for input model object in input
        if self.config.get('input_model') is not None:

            self.input_model = self.config.get('input_model')

        # Otherwise initialize input model
        else:

            self.input_model = model.get_input_model(
                self.config,
                **kwargs)

        # Get number of atomic feature vectors for scaling properties 
        self.input_n_maxatom = self.config.get('input_n_maxatom')

        # Check for graph model object in input
        if self.config.get('graph_model') is not None:

            self.graph_model = self.config.get('graph_model')

        # Otherwise initialize graph model
        else:

            self.graph_model = model.get_graph_model(
                self.config,
                **kwargs)

        # Check for output model object in input
        if self.config.get('output_model') is not None:

            self.output_model = self.config.get('output_model')

        # Otherwise initialize output model
        else:

            self.output_model = model.get_output_model(
                self.config,
                **kwargs)


    @torch.jit.export
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of PaiNN Calculator model.

        Parameters
        ----------
        batch : dict(str, torch.Tensor)
            Dictionary of input data tensors for forward pass.
            Required keys are:
                'atomic_numbers': torch.Tensor, shape(N)
                    Atomic numbers of the batch of molecules
                'positions': torch.Tensor, shape(N, 3)
                    Atomic positions of the batch of molecules
                'idx_i': torch.Tensor, shape(M)
                    Indices of atoms in pair interactions
                'idx_j': torch.Tensor, shape(M)
                    Indices of atoms in pair interactions
                'atoms_number': torch.Tensor, shape(B)
                    Number of atoms per molecule in batch
                'atoms_seg': torch.Tensor, shape(N)
                    Segment indices of atoms in batch
                'charge': torch.Tensor, shape(B)
                    Total charge of molecules in batch


        Returns
        -------

        """

        # Assign input
        atoms_number = batch['atoms_number']
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        charge = batch['charge']
        idx_seg = batch['atoms_seg']
        
        # PBC: Offset method
        pbc_offset = batch.get('pbc_offset')
        
        # PBC: Supercluster method
        atom_indices = batch.get('atom_indices')
        idx_jp = batch.get('idx_jp')
        idx_p = batch.get('idx_p')
                
        # Activate back propagation if derivatives with regard to
        # atom positions is requested.
        if self.model_gradient:
            positions.requires_grad_(True)

        # Run input model
        features, rbfs, distances, vectors, cutoff = self.input_model(
            atomic_numbers, positions, idx_i, idx_j, pbc_offset=pbc_offset)

        # Run graph model
        sfeatures, efeatures = self.graph_model(
            features, rbfs, distances, vectors, cutoff, idx_i, idx_j)
        
        # Run output model
        






    def set_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in respective model layers
        """

        ## Change unit properties for electrostatic and dispersion layers
        #if self.model_electrostatic:
            ## Synchronize total and atomic charge units
            #if model_unit_properties.get('charge') is not None:
                #model_unit_properties['atomic_charges'] = (
                    #model_unit_properties.get('charge'))
            #elif model_unit_properties.get('atomic_charges') is not None:
                #model_unit_properties['charge'] = (
                    #model_unit_properties.get('atomic_charges'))
            #else:
                #raise SyntaxError(
                    #"For electrostatic potential contribution either the"
                    #+ "model unit for the 'charge' or 'atomic_charges' must "
                    #+ "be defined!")
            #self.electrostatic_model.set_unit_properties(model_unit_properties)
        #if self.model_dispersion:
            #self.dispersion_model.set_unit_properties(model_unit_properties)

        # Store property unit labels in config file
        self.config.update({'model_unit_properties': model_unit_properties})

    def get_trainable_parameters(
        self,
        no_weight_decay: Optional[bool] = True,
    ):
        """
        Return a  dictionary of lists for different optimizer options.

        Parameters
        ----------
        no_weight_decay: bool, optional, default True
            Separate parameters on which weight decay should not be applied

        Returns
        -------
        dict(str, List)
            Dictionary of trainable model parameters. Contains 'default' entry
            for all parameters not affected by special treatment. Further
            entries are, if true, the parameter names of the input
        """

        # Trainable parameter dictionary
        trainable_parameters = {}
        trainable_parameters['default'] = []
        if no_weight_decay:
            trainable_parameters['no_weight_decay'] = []

        # Iterate over all trainable model parameters
        for name, parameter in self.named_parameters():
            # Catch all parameters to not apply weight decay on
            if no_weight_decay and 'scaling' in name.split('.')[0].split('_'):
                trainable_parameters['no_weight_decay'].append(parameter)
            elif no_weight_decay and 'dispersion_model' in name.split('.')[0]:
                trainable_parameters['no_weight_decay'].append(parameter)
            else:
                trainable_parameters['default'].append(parameter)

        return trainable_parameters

    def set_property_scaling(
        self,
        model_properties_scaling: Dict[str, List[float]]
    ):

        factor = 2.0

        # Special case: 'energy', 'atomic_energies'
        if model_properties_scaling is None:
            self.atomic_energies_scaling = torch.nn.Parameter(
                torch.tensor(
                    [1.0, 0.0],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        2
                        ).clone()
                    )
        elif (
            'energy' in self.model_properties
            and 'atomic_energies' in model_properties_scaling
        ):
            self.atomic_energies_scaling = torch.nn.Parameter(
                torch.tensor(
                    [
                        factor*model_properties_scaling['atomic_energies'][0],
                        model_properties_scaling['atomic_energies'][1]
                    ],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        len(model_properties_scaling['atomic_energies'])
                        ).clone()
                    )
        elif (
            'energy' in self.model_properties
            and 'energy' in model_properties_scaling
        ):
            self.atomic_energies_scaling = torch.nn.Parameter(
                torch.tensor(
                    [
                        factor*model_properties_scaling['energy'][0],
                        model_properties_scaling['energy'][1]
                    ],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        len(model_properties_scaling['energy'])
                        ).clone()
                    )
        else:
            self.atomic_energies_scaling = torch.nn.Parameter(
                torch.tensor(
                    [1.0, 0.0],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        2
                        ).clone()
                    )

        # Special case: 'atomic_charges'
        if model_properties_scaling is None:
            self.atomic_charges_scaling = torch.nn.Parameter(
                torch.tensor(
                    [1.0, 0.0],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        2
                        ).clone()
                    )
        elif (
            'atomic_charges' in model_properties_scaling
            and 'atomic_charges' in self.model_properties
        ):
            self.atomic_charges_scaling = torch.nn.Parameter(
                torch.tensor(
                    [
                        factor*model_properties_scaling['atomic_charges'][0],
                        model_properties_scaling['atomic_charges'][1]
                    ],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        len(model_properties_scaling['atomic_charges'])
                        ).clone()
                    )
        else:
            self.atomic_charges_scaling = torch.nn.Parameter(
                torch.tensor(
                    [1.0, 0.0],
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_maxatom,
                        2
                        ).clone()
                    )

        # Initialize scaling and shifting factors dictionary for properties
        # except atomic energies and charges
        model_scaling = {}

        # Further cases
        if model_properties_scaling is not None:
            for prop, item in model_properties_scaling.items():
                # Skip properties derived from energy and charge
                if prop in [
                    'energy', 'forces', 'hessian',
                    'atomic_charges', 'charge', 'dipole'
                ]:
                    continue
                if prop not in self.model_properties:
                    continue
                model_scaling[prop] = torch.nn.Parameter(
                    torch.tensor(
                        [factor*item[0], item[1]],
                        dtype=self.dtype, device=self.device
                        ).expand(self.input_n_maxatom, len(item)))

        # Convert model scaling to torch dictionary
        self.model_scaling = torch.nn.ParameterDict(model_scaling)

        return
