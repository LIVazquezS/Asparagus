import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import torch

from .. import model
from .. import module
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Model_PaiNN']

#======================================
# Calculator Models
#======================================

class Model_PaiNN(torch.nn.Module): 
    """
    PaiNN calculator model

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    model_properties: list(str), optional, default '['energy', 'forces']'
        Properties to predict by calculator model
    model_unit_properties: dict, optional, default {}
        Unit labels of the predicted model properties. If not defined,
        prediction results are assumed as ASE units but for during training the
        units from the reference data container are adopted.
    model_cutoff: float, optional, default 12.0
        Upper atom interaction cutoff
    model_cuton: float, optional, default None
        Lower atom pair distance to start interaction switch-off
    model_switch_range: float, optional, default 2.0
        Atom interaction cutoff switch range to switch of interaction to zero.
        If 'model_cuton' is defined, this input will be ignored.
    model_num_threads: int, optional, default 4
        Sets the number of threads used for intraop parallelism on CPU.

    """
    
    # Default arguments for graph module
    _default_args = {
        'model_properties':             ['energy', 'forces', 'dipole'],
        'model_unit_properties':        {},
        'model_cutoff':                 12.0,
        'model_cuton':                  None,
        'model_switch_range':           2.0,
        'model_num_threads':            4,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_properties':             [utils.is_string_array],
        'model_unit_properties':        [utils.is_dictionary],
        'model_cutoff':                 [utils.is_numeric],
        'model_cuton':                  [utils.is_numeric, utils.is_None],
        'model_switch_range':           [utils.is_numeric],
        'model_num_threads':            [utils.is_integer],
        }

    # Default module types of the model calculator
    _default_modules = {
        'input_type':                   'PaiNN',
        'graph_type':                   'PaiNN',
        'output_type':                  'PaiNN',
        }
    
    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_cutoff: Optional[float] = None,
        model_cuton: Optional[float] = None,
        model_switch_range: Optional[float] = None,
        model_num_threads: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PaiNN Calculator model.
        
        """

        super(Model_PaiNN, self).__init__()
        model_type = 'PaiNN'

        #############################
        # # # Check Class Input # # #
        #############################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, model),
            check_dtype=utils.get_dtype_args(self, model))

        # Update global configuration dictionary
        config.update(config_update)
        
        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')

        # Set model calculator number of threads
        if config.get('model_num_threads') is not None:
            torch.set_num_threads(config.get('model_num_threads'))

        ###################################
        # # # Check PaiNN Model Input # # #
        ###################################

        # Check model properties - Labels
        for prop in self.model_properties:
            if not utils.check_property_label(prop, return_modified=False):
                raise SyntaxError(
                    f"Model property label '{prop:s}' is not a valid property "
                    + "label! Valid property labels are:\n"
                    + list(settings._valid_properties))

        # Check model properties - Energy and energy gradient properties
        self.model_properties = list(self.model_properties)
        if any([
            prop in self.model_properties
            for prop in ['atomic_energies', 'energy']]
        ):
            self.model_energy = True
            for prop in ['atomic_energies', 'energy']:
                if prop not in self.model_properties:
                    self.model_properties.append(prop)
        else:
            self.model_energy = False
        if 'hessian' in self.model_properties:
            self.model_forces = True
            self.model_hessian = True
        elif 'forces' in self.model_properties:
            self.model_forces = True
            self.model_hessian = False
        else:
            self.model_forces = False
            self.model_hessian = False
        if self.model_forces and not self.model_energy:
            raise SyntaxError(
                f"{self.model_type:s} Model cannot predict energy gradient "
                + "properties such as forces or hessians without predicting "
                + "energies!")
        
        # Check lower cutoff switch-off range
        if self.model_cuton is None:
            if self.model_switch_range > self.model_cutoff:
                raise SyntaxError(
                    "Model cutoff switch-off range "
                    + f"({self.model_switch_range:.2f}) is larger than the "
                    + f"upper cutoff range ({self.model_cutoff:.2f})!")
            self.model_cuton = self.model_cutoff - self.model_switch_range
        elif self.model_cuton < 0.0:
            raise SyntaxError(
                "Lower atom pair cutoff distance 'model_cuton' is negative "
                + f"({self.model_cuton:.2f})!")
        elif self.model_cuton > self.model_cutoff:
            raise SyntaxError(
                "Lower atom pair cutoff distance 'model_cuton' "
                + f"({self.model_cuton:.2f}) is larger than the upper cutoff "
                + f"distance ({self.model_cutoff:.2f})!")
        else:
            self.model_switch_range = self.model_cutoff - self.model_cuton

        # Update global configuration dictionary
        config_update = {
            'model_properties': self.model_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range}
        config.update(config_update)
        
        ###############################
        # # # PaiNN Modules Setup # # #
        ###############################

        # Check for input module object in input
        if config.get('input_module') is not None:

            self.input_module = config.get('input_module')
            if hasattr(self.input_module, 'input_type'):
                self.input_type = self.input_module.input_type
            else:
                self.input_type = None

        # Otherwise initialize input module
        else:

            if config.get('input_type') is None:
                self.input_type = self._default_modules.get('input_type')
            else:
                self.input_type = config.get('input_type')
            self.input_module = module.get_input_module(
                self.input_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        # Check for graph module object in input
        if config.get('graph_module') is not None:

            self.graph_module = config.get('graph_module')
            if hasattr(self.graph_module, 'graph_type'):
                self.graph_type = self.graph_module.graph_type
            else:
                self.graph_type = None

        # Otherwise initialize graph module
        else:

            if config.get('graph_type') is None:
                self.graph_type = self._default_modules.get('graph_type')
            else:
                self.graph_type = config.get('graph_type')
            self.graph_module = module.get_graph_module(
                self.graph_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        # Check for output module object in input
        if config.get('output_module') is not None:

            self.output_module = config.get('output_module')
            if hasattr(self.output_module, 'output_type'):
                self.output_type = self.output_module.output_type
            else:
                self.output_type = None

        # Otherwise initialize output module
        else:

            if config.get('output_type') is None:
                self.output_type = self._default_modules.get('output_type')
            else:
                self.output_type = config.get('output_type')
            self.output_module = module.get_output_module(
                self.output_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        return

    def __str__(self):
        return self.model_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return model and module information
        """
        
        # Initialize info dictionary
        info = {}
        
        # Collect model info
        if hasattr(self.input_module, "get_info"):
            info = {**info, **self.input_module.get_info()}
        if hasattr(self.graph_module, "get_info"):
            info = {**info, **self.graph_module.get_info()}
        if hasattr(self.output_module, "get_info"):
            info = {**info, **self.output_module.get_info()}
        #if self.model_repulsion:
            #pass
        #if (
            #self.model_electrostatic 
            #and hasattr(self.electrostatic_module, "get_info")
        #):
            #info = {**info, **self.electrostatic_module.get_info()}
        #if (
            #self.model_dispersion
            #and hasattr(self.dispersion_module, "get_info")
        #):
            #info = {**info, **self.dispersion_module.get_info()}

        return {
            **info, 
            'model_properties': self.model_properties,
            'model_unit_properties': self.model_unit_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range,
            }

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
            Basic keys are:
                'atoms_number': torch.Tensor(n_systems)
                    Number of atoms per molecule in batch
                'atomic_numbers': torch.Tensor(n_atoms)
                    Atomic numbers of the batch of molecules
                'positions': torch.Tensor(n_atoms, 3)
                    Atomic positions of the batch of molecules
                'charge': torch.Tensor(n_systems)
                    Total charge of molecules in batch
                'idx_i': torch.Tensor(n_pairs)
                    Atom i pair index
                'idx_j': torch.Tensor(n_pairs)
                    Atom j pair index
                'sys_i': torch.Tensor(n_atoms)
                    System indices of atoms in batch
            Extra keys are:
                'pbc_offset': torch.Tensor(n_pairs)
                    Periodic boundary atom pair vector offset
                'pbc_atoms': torch.Tensor(n_atoms)
                    Primary atom indices for the supercluster approach
                'pbc_idx': torch.Tensor(n_pairs)
                    Image atom to primary atom index pointer for the atom
                    pair indices in a supercluster
                'pbc_idx_j': torch.Tensor(n_pairs)
                    Atom j pair index pointer from image atom to repsective
                    primary atom index in a supercluster

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Assign input
        atoms_number = batch['atoms_number']
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        charge = batch['charge']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        sys_i = batch['sys_i']
        
        # PBC: Offset method
        pbc_offset = batch.get('pbc_offset')
        
        # PBC: Supercluster method
        pbc_atoms = batch.get('pbc_atoms')
        pbc_idx = batch.get('pbc_idx')
        pbc_idx_j = batch.get('pbc_idx_j')
                
        # Activate back propagation if derivatives with regard to
        # atom positions is requested.
        if self.model_gradient:
            positions.requires_grad_(True)

        # Run input model
        features, distances, vectors, cutoff, rbfs = self.input_model(
            atomic_numbers, positions, idx_i, idx_j, pbc_offset=pbc_offset)

        # Run graph model
        sfeatures, efeatures = self.graph_model(
            features, distances, vectors, cutoff, rbfs, idx_i, idx_j)
        
        # Run output model
        
