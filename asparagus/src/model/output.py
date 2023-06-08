
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
# Output Model Assignment  
#======================================

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

    # Output type assignment
    output_type = config.get('output_type')

    if output_type.lower() == 'PhysNetOut'.lower():
        return Output_PhysNet(
            config,
            **kwargs)


#======================================
# Output Models
#======================================

class Output_PhysNet(torch.nn.Module): 
    """
    PhysNet Output model
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
        Initialize NNP output model.

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
        output_activation_fn: (str, object), optional,
                default 'shifted_softplus'
            Activation function
        **kwargs: dict, optional
            Additional arguments

        Returns
        -------
        callable object
            PhysNet Output model object
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

        ## Assign arguments
        #self.output_n_residual = output_n_residual
        #self.output_properties = output_properties

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
                
                prediction2 = prediction**2
                if iblock:
                    nhloss = nhloss + torch.mean(
                        prediction2/(prediction2 + last_prediction2 + 1.0e-7))
                last_prediction2 = prediction2
            
            # Save nhloss
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

















#TODO Remove when NN calculator done
class Get_Properties:
    def __init__(self,num_states,num_outputs,use_electrostatics,use_d3):
        self.num_states = num_states
        self.num_outputs = num_outputs
        self.use_electrostatics = settings._args['use_electrostatics'] #TODO: This is temporary
        self.use_d3 = settings._args['use_d3']
        self.device = settings._global_device

        if self.use_d3:
            from ..layers.grimme.d3 import edisp, d3_autoev, d3_autoang, d3_s6, d3_s8, d3_a1, d3_a2
            # Initialize variables for d3 dispersion (the way this is done,
            # positive values are guaranteed)
            if settings._args['s6'] is None:
                self.s6 = nn.Parameter(F.softplus(
                    torch.tensor(utils.softplus_inverse(d3_s6), requires_grad=True, device=self.device)))
            else:
                self.s6 = torch.tensor(settings._args['s6'], requires_grad=False, device=self.device)

            if settings._args['s8'] is None:
                self.s8 = nn.Parameter(F.softplus(
                    torch.tensor(utils.softplus_inverse(d3_s8), requires_grad=True, device=self.device)))
            else:
                self.s8 = torch.tensor(settings._args['s8'], requires_grad=False, device=self.device)

            if settings._args['a1'] is None:
                self.a1 = nn.Parameter(F.softplus(
                    torch.tensor(utils.softplus_inverse(d3_a1), requires_grad=True, device=self.device)))
            else:
                self.a1 = torch.tensor(settings._args['a1'], requires_grad=False,device=self.device)

            if settings._args['a2'] is None:
                self.a2 = nn.Parameter(F.softplus(
                    torch.tensor(utils.softplus_inverse(d3_a2), requires_grad=True, device=self.device)))
            else:
                self.a2 = torch.tensor(settings._args['a2'], requires_grad=False, device=self.device)


    @torch.jit.export
    def scaled_atomic_properties(self, output,Z):
        ''' Calculates the atomic energies, charges and distances
                    (needed if unscaled charges are wanted e.g. for loss function) '''
        #TODO: Pass this function to the output class
        #TODO: Check the lenght of the output and modify the function for multiple states/outputs
        Ea, Qa, nhloss = output
        # Apply scaling/shifting
        # TODO: Discuss how to read this from database. Remember that ALL those parameters are trainable.
        Ea = self.Escale[Z] * Ea + self.Eshift[Z]

        Qa = self.Qscale[Z] * Qa + self.Qshift[Z]

        return Ea, Qa, nhloss


    @torch.jit.export
    def energy_from_scaled_atomic_properties(
            self, Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg=None):
        ''' Calculates the energy given the scaled atomic properties (in order
            to prevent recomputation if atomic properties are calculated) '''
        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)
        # Add electrostatic and dispersion contribution to atomic energy
        if self.use_electrostatics:
            Ea = Ea + self.electrostatic_energy_per_atom(Dij, Qa, idx_i, idx_j)
        if self.use_d3:
            if self.lr_cut is not None:
                Ea = Ea + d3_autoev * edisp(Z, Dij / d3_autoang, idx_i, idx_j,
                                            s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2,
                                            cutoff=self.lr_cut / d3_autoang, device=self.device)
            else:
                Ea = Ea + d3_autoev * edisp(Z, Dij / d3_autoang, idx_i, idx_j,
                                            s6=self.s6, s8=self.s8, a1=self.a1, a2=self.a2,device=self.device)

        Ea = torch.squeeze(utils.segment_sum(Ea,batch_seg,device=self.device))

        return Ea

    @torch.jit.export
    def energy_and_forces_from_scaled_atomic_properties(
            self, Ea, Qa, Dij, Z, R, idx_i, idx_j, batch_seg=None, create_graph=True):
        ''' Calculates the energy and forces given the scaled atomic atomic
            properties (in order to prevent recomputation if atomic properties
            are calculated .
            Calculation of the forces was done following the implementation of
            spookynet. '''

        energy = self.energy_from_scaled_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad = torch.autograd.grad([torch.sum(energy)], [R], create_graph=create_graph)[0]

            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)

        return energy, forces

    @torch.jit.export
    def energy_from_atomic_properties(
            self, Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot=None, batch_seg=None):
        ''' Calculates the energy given the atomic properties (in order to
            prevent recomputation if atomic properties are calculated) '''

        if batch_seg is None:
            batch_seg = torch.zeros_like(Z, dtype=torch.int64)

            # Scale charges such that they have the desired total charge
        Qa = self.scaled_charges(Z, Qa, Q_tot, batch_seg)

        return self.energy_from_scaled_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, batch_seg)

    @torch.jit.export
    def energy_and_forces_from_atomic_properties(
            self, Ea, Qa, Dij, Z, R, idx_i, idx_j, Q_tot=None, batch_seg=None, create_graph=True):
        ''' Calculates the energy and force given the atomic properties
            (in order to prevent recomputation if atomic properties are
            calculated) '''

        energy = self.energy_from_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad = torch.autograd.grad([torch.sum(energy)], [R], create_graph=create_graph)[0]
            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)

        return energy, forces

    @torch.jit.export
    def energy(
            self, output, Z, Dij, idx_i, idx_j, Q_tot=None, batch_seg=None):
        ''' Calculates the total energy (including electrostatic
            interactions) '''

        Ea, Qa, _ = self.scaled_atomic_properties(output,Z)

        energy = self.energy_from_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)

        return energy

    @torch.jit.export
    def energy_and_forces(
            self,output, Z, R, Dij, idx_i, idx_j, Q_tot=None, batch_seg=None,create_graph=True):
        ''' Calculates the total energy and forces (including electrostatic
            interactions)'''
        Ea, Qa, _ = self.scaled_atomic_properties(output,Z)

        energy = self.energy_from_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad = torch.autograd.grad([torch.sum(energy)], [R], create_graph=create_graph)[0]
            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)
        return energy, forces

    @torch.jit.export
    def energy_and_forces_and_atomic_properties(
            self, output, Z, R, Dij, idx_i, idx_j, Q_tot=None, batch_seg=None,create_graph=True):

        ''' Calculates the total energy and forces (including electrostatic
            interactions)'''
        Ea, Qa, nhloss = self.scaled_atomic_properties(output,Z)

        energy = self.energy_from_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad = torch.autograd.grad(
                [torch.sum(energy)], [R], create_graph=create_graph)[0]
            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)

        return energy, forces, Ea, Qa, nhloss

    @torch.jit.export
    def energy_and_forces_and_charges(
            self, output,Z, R, Dij, idx_i, idx_j, Q_tot=None, batch_seg=None, create_graph=True):
        ''' Calculates the total energy and forces (including electrostatic
            interactions)'''
        Ea, Qa, nhloss = self.scaled_atomic_properties(output,Z)

        energy = self.energy_from_atomic_properties(
            Ea, Qa, Dij, Z, idx_i, idx_j, Q_tot, batch_seg)

        if idx_i.numel() > 0:  # autograd will fail if there are no distances
            grad = torch.autograd.grad(
                [torch.sum(energy)], [R], create_graph=create_graph)[0]
            if grad is not None:  # necessary for torch.jit compatibility
                forces = -grad
            else:
                forces = torch.zeros_like(R)
        else:  # if there are no distances, the forces are zero
            forces = torch.zeros_like(R)

        return energy, forces, Qa

    def scaled_charges(self, Z, Qa, Q_tot=None, batch_seg=None):
        ''' Returns scaled charges such that the sum of the partial atomic
            charges equals Q_tot (defaults to 0) '''

        if batch_seg is None:
            batch_seg = torch.zeros_like(Z)

        # Number of atoms per batch (needed for charge scaling)
        Na_helper = torch.ones_like(batch_seg, dtype=self.dtype)
        Na_per_batch = utils.segment_sum(Na_helper,batch_seg,device=self.device)

        if Q_tot is None:  # Assume desired total charge zero if not given
            Q_tot = torch.zeros_like(Na_per_batch, dtype=self.dtype)

        # Return scaled charges (such that they have the desired total charge)
        Q_correct = Q_tot - utils.segment_sum(Qa, batch_seg, device=self.device)
        Q_scaled = Qa + torch.gather((Q_correct / Na_per_batch), 0, batch_seg)

        return Q_scaled

    def _switch(self, Dij):
        ''' Switch function for electrostatic interaction (switches between
            shielded and unshielded electrostatic interaction) '''

        cut = self.sr_cut / 2
        x = Dij / cut
        x3 = x * x * x
        x4 = x3 * x
        x5 = x4 * x

        return torch.where(Dij < cut, 6 * x5 - 15 * x4 + 10 * x3, torch.ones_like(Dij))

    def electrostatic_energy_per_atom(self, Dij, Qa, idx_i, idx_j):
        ''' Calculates the electrostatic energy per atom for very small
            distances, the 1/r law is shielded to avoid singularities '''

        # Gather charges
        Qi = torch.gather(Qa, 0, idx_i)
        Qj = torch.gather(Qa, 0, idx_j)

        # Calculate variants of Dij which we need to calculate
        # the various shielded/non-shielded potentials
        DijS = torch.sqrt(Dij * Dij + 1.0)  # shielded distance

        # Calculate value of switching function
        switch = self._switch(Dij)  # normal switch
        cswitch = 1.0 - switch  # complementary switch

        # Calculate shielded/non-shielded potentials
        if self.lr_cut is None:  # no non-bonded cutoff

            Eele_ordinary = 1.0 / Dij  # ordinary electrostatic energy
            Eele_shielded = 1.0 / DijS  # shielded electrostatic energy

            # Combine shielded and ordinary interactions and apply prefactors
            Eele = self.kehalf * Qi * Qj * (
                    cswitch * Eele_shielded + switch * Eele_ordinary)

        else:  # with non-bonded cutoff

            cut = self.lr_cut
            cut2 = self.lr_cut * self.lr_cut

            Eele_ordinary = 1.0 / Dij + Dij / cut2 - 2.0 / cut
            Eele_shielded = 1.0 / DijS + DijS / cut2 - 2.0 / cut

            # Combine shielded and ordinary interactions and apply prefactors
            Eele = self.kehalf * Qi * Qj * (
                    cswitch * Eele_shielded + switch * Eele_ordinary)
            Eele = torch.where(Dij <= cut, Eele, torch.zeros_like(Eele))

        Eele_f = utils.segment_sum(Eele,idx_i,device=self.device)

        return Eele_f
