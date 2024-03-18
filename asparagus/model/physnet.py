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

__all__ = ['Model_PhysNet']

#======================================
# Calculator Models
#======================================

class Model_PhysNet(torch.nn.Module): 
    """
    PhysNet Calculator model

    Parameters
    ----------
    config: (str, dict, object)
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
    model_repulsion: bool, optional, default False
        Use close-range atom repulsion model.
    model_electrostatic: bool, optional, default True
        Use electrostatic potential between atomic charges for energy
        prediction.
    model_dispersion: bool, optional, default True
        Use Grimme's D3 dispersion model for energy prediction.
    model_dispersion_trainable: bool, optional, default True
        If True, empirical parameter in the D3 dispersion model are
        trainable. If False, empirical parameter are fixed to default

    """
    
    # Default arguments for graph module
    _default_args = {
        'model_properties':             ['energy', 'forces'],
        'model_unit_properties':        {},
        'model_cutoff':                 12.0,
        'model_cuton':                  None,
        'model_switch_range':           2.0,
        'model_repulsion':              False,
        'model_electrostatic':          True,
        'model_dispersion':             True,
        'model_dispersion_trainable':   True,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_properties':             [utils.is_string_array],
        'model_unit_properties':        [utils.is_dictionary],
        'model_cutoff':                 [utils.is_numeric],
        'model_cuton':                  [utils.is_numeric, utils.is_None],
        'model_switch_range':           [utils.is_numeric],
        'model_repulsion':              [utils.is_bool],
        'model_electrostatic':          [utils.is_bool],
        'model_dispersion':             [utils.is_bool],
        'model_dispersion_trainable':   [utils.is_bool],
        }

    # Default module types of the model calculator
    _default_modules = {
        'input_type':                   'PhysNet',
        'graph_type':                   'PhysNet',
        'output_type':                  'PhysNet',
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
        model_repulsion: Optional[bool] = None,
        model_electrostatic: Optional[bool] = None,
        model_dispersion: Optional[bool] = None,
        model_dispersion_trainable: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize PhysNet Calculator model.
        
        """

        super(Model_PhysNet, self).__init__()
        model_type = 'PhysNet'

        ##########################################
        # # # Check PhysNet Calculator Input # # #
        ##########################################
        
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

        ##########################################
        # # # Check PhysNet Model Properties # # #
        ##########################################

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

        # Check model properties - Atomic charges and derivatives
        if 'dipole' in self.model_properties:
            self.model_atomic_charges = True
            self.model_dipole = True
            if 'atomic_charges' not in self.model_properties:
                self.model_properties.append('atomic_charges')
        elif 'atomic_charges' in self.model_properties:
            self.model_atomic_charges = True
            self.model_dipole = False
        else:
            self.model_atomic_charges = False
            self.model_dipole = False

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

        #################################
        # # # PhysNet Modules Setup # # #
        #################################

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

        # Assign atom repulsion module
        if self.model_repulsion:
            pass

        # Assign electrostatic interaction module
        if self.model_electrostatic and self.model_energy:

            # Get electrostatic point charge model calculator
            self.electrostatic_module = module.PC_shielded_electrostatics(
                cutoff = self.model_cutoff,
                cutoff_short_range=config.get('input_radial_cutoff'),
                unit_properties=self.model_unit_properties,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        elif self.model_electrostatic:
            
            raise SyntaxError(
                "Electrostatic energy contribution is requested without "
                + "having 'energy' assigned as model property!")

        # Assign dispersion interaction module
        if self.model_dispersion and self.model_energy:
            
            # Grep dispersion correction parameters
            d3_s6 = config.get("model_dispersion_d3_s6")
            d3_s8 = config.get("model_dispersion_d3_s8")
            d3_a1 = config.get("model_dispersion_d3_a1")
            d3_a2 = config.get("model_dispersion_d3_a2")
            
            # Get Grimme's D3 dispersion model calculator
            self.dispersion_module = module.D3_dispersion(
                self.model_cutoff,
                cuton=self.model_cuton,
                unit_properties=self.model_unit_properties,
                d3_s6=d3_s6,
                d3_s8=d3_s8,
                d3_a1=d3_a1,
                d3_a2=d3_a2,
                trainable=self.model_dispersion_trainable,
                device=self.device,
                dtype=self.dtype)

        elif self.model_dispersion:
            
            raise SyntaxError(
                "Dispersion energy contribution is requested without "
                + "having 'energy' assigned as model property!")

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
        if self.model_repulsion:
            pass
        if (
            self.model_electrostatic 
            and hasattr(self.electrostatic_module, "get_info")
        ):
            info = {**info, **self.electrostatic_module.get_info()}
        if (
            self.model_dispersion
            and hasattr(self.dispersion_module, "get_info")
        ):
            info = {**info, **self.dispersion_module.get_info()}

        return {
            **info, 
            'model_properties': self.model_properties,
            'model_unit_properties': self.model_unit_properties,
            'model_cutoff': self.model_cutoff,
            'model_cuton': self.model_cuton,
            'model_switch_range': self.model_switch_range,
            'model_repulsion': self.model_repulsion,
            'model_electrostatic': self.model_electrostatic,
            'model_dispersion': self.model_dispersion,
            'model_dispersion_trainable': self.model_dispersion_trainable,
            }

    def set_property_scaling(
        self, 
        scaling_parameter: Optional[Dict[str, List[float]]] = None,
        atomic_energies_shifts: Optional[Dict[Union[int, str], float]] = None
    ):
        """
        Prepare property scaling factor and shift terms and set atomic type
        energies shift.
        
        """
        
        # Set property scaling factors and shift terms
        if scaling_parameter is not None:
            self.output_module.set_property_scaling(scaling_parameter)

        # Set atomic type energies shift
        if atomic_energies_shifts is not None:
            self.output_module.set_atomic_energies_shift(
                atomic_energies_shifts)

        return

    def set_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in respective model layers
        
        Parameter
        ---------
        model_unit_properties: dict
            Unit labels of the predicted model properties

        """

        # Change unit properties for electrostatic and dispersion layers
        if self.model_electrostatic:
            # Synchronize total and atomic charge units
            if model_unit_properties.get('charge') is not None:
                model_unit_properties['atomic_charges'] = (
                    model_unit_properties.get('charge'))
            elif model_unit_properties.get('atomic_charges') is not None:
                model_unit_properties['charge'] = (
                    model_unit_properties.get('atomic_charges'))
            else:
                raise SyntaxError(
                    "For electrostatic potential contribution either the"
                    + "model unit for the 'charge' or 'atomic_charges' must "
                    + "be defined!")
            self.electrostatic_module.set_unit_properties(
                model_unit_properties)
        if self.model_dispersion:
            self.dispersion_module.set_unit_properties(model_unit_properties)

        return

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
            elif no_weight_decay and 'shift' in name.split('.')[0].split('_'):
                trainable_parameters['no_weight_decay'].append(parameter)
            elif no_weight_decay and 'dispersion_model' in name.split('.')[0]:
                trainable_parameters['no_weight_decay'].append(parameter)
            else:
                trainable_parameters['default'].append(parameter)

        return trainable_parameters

    @torch.jit.export
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of PhysNet Calculator model.

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
                'atoms_seg': torch.Tensor(n_atoms)
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
    
        # Activate back propagation if derivatives with regard to atom positions 
        # is requested.
        if self.model_forces:
            positions.requires_grad_(True)

        # Run input model
        features, distances, cutoffs, rbfs = self.input_module(
            atomic_numbers, positions, idx_i, idx_j, pbc_offset=pbc_offset)

        # Compute descriptors
        descriptors = cutoffs*rbfs

        # PBC: Supercluster approach - Point from image atoms to primary atoms
        if idx_p is not None:
            idx_i = pbc_idx[idx_i]
            idx_j = pbc_idx[pbc_idx_j]

        # Run graph model
        features_list = self.graph_module(features, descriptors, idx_i, idx_j)

        # Run output model
        output = self.output_module(
            features_list, 
            atomic_numbers=atomic_numbers)
        
        # Add repulsion model contribution
        if self.model_repulsion:
            pass

        # Add dispersion model contributions
        if self.model_dispersion:
            output['atomic_energies'] = (
                output['atomic_energies']
                + self.dispersion_module(
                    atomic_numbers, distances, idx_i, idx_j))

        # Scale atomic charges to ensure correct total charge
        if self.model_atomic_charges:
            charge_deviation = (
                charge - utils.segment_sum(
                    output['atomic_charges'], idx_seg, device=self.device)
                / atoms_number
                )
            output['atomic_charges'] = (
                output['atomic_charges'] + charge_deviation[idx_seg])

        # Add electrostatic model contribution
        if self.model_electrostatic:
            # Apply electrostatic model
            output['atomic_energies'] = (
                output['atomic_energies']
                + self.electrostatic_module(
                    output['atomic_charges'], 
                    distances, idx_i, idx_j))

        # Compute property - Energy
        if self.model_energy:
            output['energy'] = torch.squeeze(
                utils.segment_sum(
                    output['atomic_energies'], sys_i, device=self.device)
                )

        # Compute gradients and Hessian if demanded
        if self.model_forces:

            gradient = torch.autograd.grad(
                torch.sum(output['energy']),
                positions,
                create_graph=True)[0]

            # Avoid crashing if forces are none
            if gradient is not None:
                output['forces'] = -gradient
            else:
                logger.warning(
                    "WARNING:\nError in force calculation "
                    + "(backpropagation)!")
                output['forces'] = torch.zeros_like(positions)

            if self.model_hessian:
                hessian = output['energy'].new_zeros(
                    (gradient.size(0), gradient.size(0)))
                for ig in range(gradient.size(0)):
                    hessian_ig = torch.autograd.grad(
                        [gradient[ig]],
                        positions,
                        retain_graph=(ig < gradient.size(0)))[0]
                    if hessian_ig is not None:
                        hessian[ig] = hessian_ig.view(-1)
                output['hessian'] = hessian

        # Compute molecular dipole of demanded
        if self.model_dipole:
            if pbc_atoms is None:
                output['dipole'] = utils.segment_sum(
                    output['atomic_charges'][..., None]*positions,
                    sys_i, device=self.device).reshape(-1, 3)
            else:
                output['dipole'] = utils.segment_sum(
                    output['atomic_charges'][..., None]
                    *positions[pbc_atoms],
                    sys_i, device=self.device).reshape(-1, 3)

        #print('properties predicted: ', output.keys())
        #print('energy: ', output['energy'])
        #print('forces: ', output['forces'])
        #print('atomic charges: ', output['atomic_charges'])
        #print('dipole: ', output['dipole'])

        return output

