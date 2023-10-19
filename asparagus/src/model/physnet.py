
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

__all__ = ['Calculator_PhysNet']

# ======================================
# Calculator Models
# ======================================


class Calculator_PhysNet(torch.nn.Module):
    """
    PhysNet Calculator model
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        model_properties: Optional[List[str]] = None,
        model_unit_properties: Optional[Dict[str, str]] = None,
        model_descriptor_cutoff: Optional[float] = None,
        model_interaction_cutoff: Optional[float] = None,
        model_cutoff_width: Optional[float] = None,
        model_repulsion: Optional[bool] = None,
        model_electrostatic: Optional[bool] = None,
        model_dispersion: Optional[bool] = None,
        model_dispersion_trainable: Optional[bool] = None,
        model_properties_scaling: Optional[Dict[str, List[float]]] = None,
        **kwargs
    ):
        """
        Initialize NNP Calculator model.

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
        model_cutoff_width: float, optional, default 2
            Cutoff switch width to converge zero in the range from
            cutoff to cutoff - width.
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
        model_properties_scaling: dict(str, list), optional, default None
            Property scaling factor and shift term initially predicted
            by reference data distribution to improve convergence in NN fit.
        **kwargs: dict, optional
            Additional arguments

        Returns
        -------
        callable object
            PhysNet Calculator object for training
        """

        super(Calculator_PhysNet, self).__init__()

        ##########################################
        # # # Check PhysNet Calculator Input # # #
        ##########################################

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

        ###################################
        # # # Prepare PhysNet Modules # # #
        ###################################

        # Check for input model object in input
        if self.config.get('input_model') is not None:

            self.input_model = self.config.get('input_model')

        # Otherwise initialize input model
        else:

            self.input_model = model.get_input_model(
                self.config,
                **kwargs)

        # Check for graph model object in input
        if config.get('graph_model') is not None:

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

        ######################################
        # # # Prepare PhysNet Calculator # # #
        ######################################

        # Calculator class type
        self.model_type = 'PhysNet'

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

        # Check for atomic charge derivatives in prediction list
        if 'dipole' in self.model_properties:
            self.model_dipole = True
            if 'atomic_charges' not in self.model_properties:
                raise ValueError(
                    "PhysNet model cannot provide molecular dipole "
                    + "without the prediction of atomic charges!")
        elif 'atomic_charges' in self.model_properties:
            self.model_dipole = True
        else:
            self.model_dipole = False

        # Get length of atomic feature vector
        self.input_n_atombasis = self.config.get('input_n_atombasis')
        self.input_cutoff_descriptor = self.config.get(
            'input_cutoff_descriptor')

        # Check cutoffs
        if self.model_interaction_cutoff > self.input_cutoff_descriptor:
            self.model_cutoff_split = True
        elif self.model_interaction_cutoff < self.input_cutoff_descriptor:
            raise ValueError(
                "The interaction cutoff distance 'model_interaction_cutoff' "
                + f"({self.model_interaction_cutoff:.2f}) "
                + "must be larger than or equal the descriptor range "
                + "'input_cutoff_descriptor' "
                + f"({self.input_cutoff_descriptor:.2f})!")
        else:
            self.model_cutoff_split = False

        # Check cutoff width
        if self.model_cutoff_width == 0.0:
            logger.warning(
                "WARNING:\n The interaction cutoff width 'model_cutoff_width'"
                + "is zero which might lead to indifferentiable potential"
                + "at the interaction cutoff at "
                + f"{self.model_interaction_cutoff:.2f}!")
        elif self.model_cutoff_width < 0.0:
            raise ValueError(
                "The interaction cutoff width 'model_cutoff_width' "
                + f"({self.model_cutoff_width:.2f}) "
                + "must be larger or equal zero!")

        # Assign atom repulsion parameters
        if self.model_repulsion:
            pass

        # Initialize electrostatic interaction model
        if self.model_electrostatic and self.model_dipole:

            # Get electrostatic point charge model calculator
            self.electrostatic_model = layers.PC_shielded_electrostatics(
                self.model_cutoff_split,
                self.input_cutoff_descriptor,
                self.model_interaction_cutoff,
                unit_properties=self.model_unit_properties,
                switch_fn="Poly6",
                device=self.device,
                dtype=self.dtype)

        elif self.model_electrostatic and not self.model_dipole:

            logger.warning(
                "WARNING:\n"
                + "PhysNet model cannot provide electrostatic contribution "
                + "to the atomic energies without the prediction of "
                + "atomic charges or dipole moment via atomic charges!")

        # Initialize dispersion model
        if self.model_dispersion:

            # Check dispersion correction parameters
            d3_s6 = self.config.get("model_dispersion_d3_s6")
            d3_s8 = self.config.get("model_dispersion_d3_s8")
            d3_a1 = self.config.get("model_dispersion_d3_a1")
            d3_a2 = self.config.get("model_dispersion_d3_a2")

            # Get Grimme's D3 dispersion model calculator
            self.dispersion_model = layers.D3_dispersion(
                self.model_interaction_cutoff,
                self.model_cutoff_width,
                unit_properties=self.model_unit_properties,
                d3_s6=d3_s6,
                d3_s8=d3_s8,
                d3_a1=d3_a1,
                d3_a2=d3_a2,
                trainable=self.model_dispersion_trainable,
                device=self.device,
                dtype=self.dtype)

        # initialize property scaling parameter
        self.set_property_scaling(self.model_properties_scaling)

        return

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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        self.input_n_atombasis,
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
                        ).expand(self.input_n_atombasis, len(item)))

        # Convert model scaling to torch dictionary
        self.model_scaling = torch.nn.ParameterDict(model_scaling)

        return

    def set_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in respective model layers
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
            self.electrostatic_model.set_unit_properties(model_unit_properties)
        if self.model_dispersion:
            self.dispersion_model.set_unit_properties(model_unit_properties)

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

    @torch.jit.export
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # Assign input
        atoms_number = batch['atoms_number']
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        charge = batch['charge']
        idx_seg = batch['atoms_seg']
        pbc_offset = batch.get('pbc_offset')

        # Activate back propagation if derivatives with regard to
        # atom positions is requested.
        if self.model_gradient:
            positions.requires_grad_(True)

        # Run input model
        features, descriptors, distances = self.input_model(
            atomic_numbers, positions, idx_i, idx_j, pbc_offset=pbc_offset)

        # Run graph model
        messages = self.graph_model(features, descriptors, idx_i, idx_j)

        # Run output model
        output = self.output_model(messages)

        # Scale atomic energies by fit parameter
        scale, shift = self.atomic_energies_scaling[atomic_numbers].T
        output['atomic_energies'] = output['atomic_energies']*scale + shift

        # Add repulsion model contribution
        if self.model_repulsion:
            pass

        # Add dispersion model contribution
        if self.model_dispersion:
            output['atomic_energies'] = (
                output['atomic_energies']
                + self.dispersion_model(
                    atomic_numbers, distances, idx_i, idx_j))

        # Add electrostatic model contribution
        if self.model_electrostatic:

            # Scale atomic charges by fit parameter
            scale, shift = self.atomic_charges_scaling[atomic_numbers].T
            output['atomic_charges'] = output['atomic_charges']*scale + shift

            # Scale atomic charges to ensure correct total charge
            charge_deviation = charge - utils.segment_sum(
                output['atomic_charges'], idx_seg, device=self.device)
            output['atomic_charges'] = (
                output['atomic_charges']
                + (charge_deviation/atoms_number)[idx_seg])

            # Apply electrostatic model
            output['atomic_energies'] = (
                output['atomic_energies']
                + self.electrostatic_model(
                    output['atomic_charges'],
                    distances, idx_i, idx_j))

        # Compute system energies
        output['energy'] = torch.squeeze(
            utils.segment_sum(
                output['atomic_energies'], idx_seg, device=self.device)
            )

        # Charges are missing??
        if 'atomic_charges' in output:
            output['charge'] = utils.segment_sum(
                output['atomic_charges'], idx_seg, device=self.device)

        # Compute gradients and Hessian if demanded
        if (self.model_forces or self.model_hessian) and idx_i.numel():

            gradient = torch.autograd.grad(
                torch.sum(output['energy']),
                positions,
                create_graph=True)[0]

            if self.model_forces:
                # Avoid crashing if forces are none
                if gradient is not None:
                    output['forces'] = -gradient
                else:
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
            output['dipole'] = utils.segment_sum(
                output['atomic_charges'][..., None]*positions,
                idx_seg, device=self.device).reshape(-1, 3)

        # Apply output scaling and shift
        for prop, item in self.model_scaling.items():
            prop_scale, prop_shift = item[atomic_numbers].T
            output[prop] = output[prop]*prop_scale + prop_shift

        return output
