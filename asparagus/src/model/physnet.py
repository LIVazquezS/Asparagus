
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

        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device

        ###################################
        # # # Prepare PhysNet Modules # # #
        ###################################

        # Check for input model object in input
        if config.get('input_model') is not None:

            self.input_model = config.get('input_model')

        # Otherwise initialize input model
        else:

            self.input_model = model.get_input_model(
                config,
                **kwargs)

        # Check for graph model object in input
        if config.get('graph_model') is not None:

            self.graph_model = config.get('graph_model')

        # Otherwise initialize graph model
        else:

            self.graph_model = model.get_graph_model(
                config,
                **kwargs)

        # Check for output model object in input
        if config.get('output_model') is not None:

            self.output_model = config.get('output_model')

        # Otherwise initialize output model
        else:

            self.output_model = model.get_output_model(
                config,
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
                    "PhysNet model cannot provide molecular dipole " +
                    "without the prediction of atomic charges!")
        else:
            self.model_dipole = False

        # Get length of atomic feature vector
        self.input_n_atombasis = config.get('input_n_atombasis')
        self.input_cutoff_descriptor = config.get('input_cutoff_descriptor')

        # Check cutoffs
        if self.model_interaction_cutoff > self.input_cutoff_descriptor:
            self.model_cutoff_split = True
        elif self.model_interaction_cutoff < self.input_cutoff_descriptor:
            raise ValueError(
                f"The interaction cutoff distance 'model_interaction_cutoff' " +
                f"({self.model_interaction_cutoff:.2f})" +
                f"must be larger than or equal the descriptor range " +
                f"'input_cutoff_descriptor' " +
                f"({self.input_cutoff_descriptor:.2f})!")
        else:
            self.model_cutoff_split = False

        # Assign atom repulsion parameters
        if self.model_repulsion:
            pass

        # Initialize electrostatic interaction model
        if self.model_electrostatic:

            # Get electrostatic point charge model calculator
            self.electrostatic_model = layers.PC_shielded_electrostatics(
                self.model_cutoff_split,
                self.input_cutoff_descriptor,
                self.model_interaction_cutoff,
                switch_fn="Poly6",
                device=self.device,
                dtype=self.dtype)

        # Initialize dispersion model
        if self.model_dispersion:

            # Check dispersion correction parameters
            d3_s6 = config.get("model_dispersion_d3_s6")
            d3_s8 = config.get("model_dispersion_d3_s8")
            d3_a1 = config.get("model_dispersion_d3_a1")
            d3_a2 = config.get("model_dispersion_d3_a2")

            # Get Grimme's D3 dispersion model calculator
            self.dispersion_model = layers.D3_dispersion(
                self.model_interaction_cutoff,
                self.model_cutoff_width,
                d3_s6=d3_s6,
                d3_s8=d3_s8,
                d3_a1=d3_a1,
                d3_a2=d3_a2,
                trainable=self.model_dispersion_trainable,
                device=self.device,
                dtype=self.dtype)

        # Initialize scaling and shifting factors
        self.model_scaling = {}

        # Special case: 'energy', 'atomic_energies'
        if ('energy' in self.model_properties_scaling.keys()
            and not 'atomic_energies' in self.model_properties_scaling.keys()
            and 'energy' in self.model_properties
            ):
            self.model_scaling['atomic_energies'] = torch.nn.Parameter(
                torch.tensor(
                    self.model_properties_scaling['energy'], 
                    dtype=self.dtype,
                    device=self.device
                    ).expand(
                        self.input_n_atombasis, 
                        len(self.model_properties_scaling['energy'])))

        # Special case: 'atomic_charges'
        if (not 'atomic_charges' in self.model_properties_scaling.keys()
            and 'atomic_charges' in self.model_properties
            ):
            self.model_scaling['atomic_charges'] = torch.nn.Parameter(
                torch.tensor(
                    [1.0, 0.0], dtype=self.dtype, device=self.device
                    ).expand(self.input_n_atombasis, 2))

        # Other cases
        for prop, item in self.model_properties_scaling.items():
            # Skip certain properties
            if prop in ['energy', 'forces', 'hessian', 'charge', 'dipole']:
                continue
            if prop not in self.model_properties:
                continue
            self.model_scaling[prop] = torch.nn.Parameter(
                torch.tensor(
                    item, dtype=self.dtype, device=self.device
                    ).expand(self.input_n_atombasis, len(item)))

        return


    @torch.jit.export
    def forward(
        self,
        batch: Dict[str, torch.Tensor]
        #atoms_number: torch.Tensor,
        #atomic_numbers: torch.Tensor,
        #positions: torch.Tensor,
        #idx_i: torch.Tensor,
        #idx_j: torch.Tensor,
        #charge: torch.Tensor,
        #idx_seg: torch.Tensor,
        #pbc_offset: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # Assign input
        atoms_number = batch['atoms_number']
        atomic_numbers = batch['atomic_numbers']
        positions = batch['positions']
        idx_i = batch['idx_i']
        idx_j = batch['idx_j']
        charge = batch['charge']
        idx_seg = batch['atoms_seg']
        pbc_offset = batch['pbc_offset']
    
        # Activate back propagation if derivatives with regard to atom positions 
        # is requested.
        if self.model_gradient:
            positions.requires_grad_(True)

        # Run input model
        features, descriptors, distances = self.input_model(
            atomic_numbers, positions, idx_i, idx_j, pbc_offset=pbc_offset)

        # Run graph model
        messages = self.graph_model(features, descriptors, idx_i, idx_j)

        # Run output model
        output = self.output_model(messages)
        
        # Apply output scaling and shift
        for prop, item in self.model_scaling.items():
            prop_scale, prop_shift = item[atomic_numbers].T
            output[prop] = output[prop]*prop_scale + prop_shift

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

        #print('properties predicted: ', output.keys())
        #print('energy: ', output['energy'])
        #print('forces: ', output['forces'])
        #print('atomic charges: ', output['atomic_charges'])
        #print('dipole: ', output['dipole'])

        return output