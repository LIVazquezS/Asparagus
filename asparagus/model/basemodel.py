import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np 

import ase
import torch

from asparagus import model
from asparagus import module
from asparagus import settings
from asparagus import utils

__all__ = ['Model_PhysNet']

#======================================
# Calculator Models
#======================================

class BaseModel(torch.nn.Module): 
    """
    Asparagus calculator base model

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    _default_model_properties = ['energy', 'forces']

    _supported_model_properties = [
        'energy',
        'atomic_energies',
        'forces']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        **kwargs
    ):
        """
        Initialize BaseModel Calculator
        """

        super(BaseModel, self).__init__()
        self.model_type = 'BaseModel'
        
        # Initialize loaded checkpoint flag
        self.checkpoint_loaded = False

        return

    def __str__(self):
        return self.model_type

    def get_info(self) -> Dict[str, Any]:
        """
        Return model and module information
        """
        return {}

    def check_model_properties(
        self,
        config: settings.Configuration,
        model_properties: List[str],
    ) -> List[str]:
        """
        Check model property input.
        
        Parameters
        ----------
        config: settings.Configuration
            Asparagus settings.Configuration class object of global parameter
        model_properties: list(str)
            Properties to predict by calculator model

        Returns
        ----------
        list(str)
            Checked property labels

        """
        
        # Check model properties - Selection
        if model_properties is None:
            # If no model properties are defined but data properties are,
            # adopt all supported data properties as model properties,
            # else adopt default model properties
            if config.get('data_properties') is None:
                model_properties = self._default_model_properties
            else:
                model_properties = []
                for prop in config.get('data_properties'):
                    if prop in self._supported_model_properties:
                        model_properties.append(prop)

        # Check model properties - Labels
        for prop in model_properties:
            if not utils.check_property_label(prop, return_modified=False):
                raise SyntaxError(
                    f"Model property label '{prop:s}' is not a valid property "
                    + "label! Valid property labels are:\n"
                    + str(list(settings._valid_properties)))
        
        return list(model_properties)

    def set_model_energy_properties(
        self,
        model_properties: List[str],
        model_energy_properties: 
            Optional[List[str]] = ['atomic_energies', 'energy'],
    ) -> List[str]:
        """
        Set model energy property parameters.
        
        Parameters
        ----------
        model_properties: list(str)
            Properties to predict by calculator model
        model_energy_properties: list(str)
            Model energy related properties

        Returns
        ----------
        list(str)
            Checked property labels

        """

        # Check model properties - Energy and energy gradient properties
        if any([
            prop in model_properties
            for prop in model_energy_properties]
        ):
            self.model_energy = True
            for prop in model_energy_properties:
                if prop not in model_properties:
                    model_properties.append(prop)
        else:
            self.model_energy = False
        if 'hessian' in model_properties:
            self.model_forces = True
            self.model_hessian = True
        elif 'forces' in model_properties:
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

        return model_properties

    def check_model_property_units(
        self,
        model_properties: List[str],
        model_unit_properties: Dict[str, str],
        model_default_properties: 
            Optional[List[str]] = ['positions', 'charge'],
    ) -> Dict[str, str]:
        """
        Check model property units input.
        
        Parameters
        ----------
        model_properties: list(str)
            Properties to predict by calculator model
        model_unit_properties: dict
            Unit labels of the predicted model properties.
        model_default_properties: list(str), optional, 
                default ['positions', 'charge']
            Default properties where default unit settings are adopted even
            if not defined as model property.

        Returns
        ----------
        dict(str, str)
            Checked unit labels of the predicted model properties.

        """

        # Check property units input
        if model_unit_properties is None:
            model_unit_properties = {}

        # Initialize checked property units dictionary
        checked_unit_properties = {}

        # Check if default properties units are defined
        for prop in model_default_properties:
            if prop not in model_unit_properties:
                checked_unit_properties[prop] = (
                    settings._default_units[prop])
                self.logger.info(
                    f"Unit for property '{prop}' is set to the default unit "
                    + f"'{settings._default_units[prop]}'!")
            else:
                checked_unit_properties[prop] = (
                    model_unit_properties[prop])

        # Check if all model property units are defined
        for prop in model_properties:
            if prop not in model_unit_properties:
                # Check if a related property unit is defined
                for rel_props in settings._related_unit_properties:
                    if prop in rel_props:
                        for rprop in rel_props:
                            if rprop in model_unit_properties:
                                checked_unit_properties[prop] = (
                                    model_unit_properties[rprop])
                                break
                # Else, take default
                self.logger.warning(
                    f"No unit defined for property '{prop}'!\n"
                    + f"Default unit of '{settings._default_units[prop]}' "
                    + "will be used.")
                checked_unit_properties[prop] = (
                    settings._default_units[prop])
            else:
                checked_unit_properties[prop] = (
                    model_unit_properties[prop])

        return checked_unit_properties

    def check_cutoff_ranges(
        self,
        model_cutoff: float,
        model_cuton: float,
        model_switch_range: float,
    ) -> (float, float, float):
        """
        Check model cutoff range option.
        
        Parameters
        ----------
        model_cutoff: float
            Upper atom interaction cutoff
        model_cuton: float
            Lower atom pair distance to start interaction switch-off
        model_switch_range: float
            Atom interaction cutoff switch range to switch of interaction to 
            zero. If 'model_cuton' is defined, this input will be ignored.

        Returns
        ----------
        float
            Model start interaction switch-off range
        float
            Model interaction cutoff switch range

        """
        
        # Check lower cutoff switch-off range
        if model_cuton is None:
            if model_switch_range > model_cutoff:
                raise SyntaxError(
                    "Model cutoff switch-off range "
                    + f"({model_switch_range:.2f}) is larger than the "
                    + f"upper cutoff range ({model_cutoff:.2f})!")
            model_cuton = model_cutoff - model_switch_range
        elif model_cuton > model_cutoff:
            message = (
                    "Lower atom pair cutoff distance 'model_cuton' "
                    + f"({model_cuton:.2f}) is larger than the upper cutoff "
                    + f"distance ({model_cutoff:.2f})!")
            if model_switch_range is None:
                raise SyntaxError(message)
            else:
                model_cuton = model_cutoff - model_switch_range
                self.logger.warning(
                    f"{message:s}\n"
                    + "Lower atom pair cutoff distance is changed switched "
                    + f"to '{model_cuton:.2f}'.")
        else:
            model_switch_range = model_cutoff - model_cuton
        if model_cuton < 0.0:
            raise SyntaxError(
                "Lower atom pair cutoff distance 'model_cuton' is negative "
                + f"({model_cuton:.2f})!")
        
        return model_cuton, model_switch_range

    def base_modules_setup(
        self,
        config: settings.Configuration,
        **kwargs
    ):
        """
        Setup model calculator base modules input, graph and output
        
        Parameters
        ----------
        config: settings.Configuration
            Asparagus settings.Configuration class object of global parameter
        **kwargs
            Base module options

        """

        # Check for input module object in configuration input,
        # otherwise initialize input module
        if config.get('input_module') is not None:

            input_module = config.get('input_module')

        else:

            if config.get('input_type') is None:
                input_type = self._default_modules.get('input_type')
            else:
                input_type = config.get('input_type')
            input_module = module.get_input_module(
                input_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        # Check for graph module object in configuration input,
        # otherwise initialize graph module
        if config.get('graph_module') is not None:

            graph_module = config.get('graph_module')

        else:

            if config.get('graph_type') is None:
                graph_type = self._default_modules.get('graph_type')
            else:
                graph_type = config.get('graph_type')
            graph_module = module.get_graph_module(
                graph_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)

        # Check for output module object in input,
        # otherwise initialize output module
        if config.get('output_module') is not None:

            output_module = config.get('output_module')

        else:

            if config.get('output_type') is None:
                output_type = self._default_modules.get('output_type')
            else:
                output_type = config.get('output_type')
            output_module = module.get_output_module(
                output_type,
                config=config,
                device=self.device,
                dtype=self.dtype,
                **kwargs)
        
        return input_module, graph_module, output_module

    def get_scaleable_properties(
        self
    ) -> List[str]:
        """
        Return list of properties which are scaled by a scaling factor and
        shift term, which initial guess are derived from reference data.

        Returns
        -------
        list(str)
            Scalable model properties

        """
        return self.output_module.output_properties

    def set_property_scaling(
        self,
        property_scaling: Optional[
            Dict[str, Union[List[float], Dict[int, List[float]]]]] = None,
        set_shift_term: Optional[bool] = True,
        set_scaling_factor: Optional[bool] = True,
    ):
        """
        Set property scaling factor and shift terms and set atomic type
        energies shift.

        Parameters
        ----------
        property_scaling: dict(str, (list(float), dict(int, float)), optional,
                default None
            Model property scaling parameter to shift and scale output module
            results
        set_shift_term: bool, optional, default True
            If True, set or update the shift term. Else, keep previous
            value.
        set_scaling_factor: bool, optional, default True
            If True, set or update the scaling factor. Else, keep previous
            value.

        """
        
        # Set property scaling factors and shift terms
        if (
            property_scaling is not None
            and hasattr(self.output_module, "set_property_scaling")
        ):
            self.output_module.set_property_scaling(
                property_scaling,
                set_shift_term=set_shift_term,
                set_scaling_factor=set_scaling_factor)

        return

    def get_property_scaling(
        self,
    ) -> Dict[str, Union[List[float], Dict[int, List[float]]]]:
        """
        Get current property scaling factor and shift term dictionary.

        Returns
        -------
        dict(str, (list(float), dict(int, float))
            Current model property scaling parameter to shift and scale output
            module results

        """

        # Get property scaling factors and shift terms
        if hasattr(self.output_module, "get_property_scaling"):
            property_scaling = self.output_module.get_property_scaling()
        else:
            property_scaling = {}

        return property_scaling

    def set_model_unit_properties(
        self,
        model_unit_properties: Dict[str, str],
    ):
        """
        Set or change unit property parameter in model layers or modules

        Parameters
        ----------
        model_unit_properties: dict
            Dictionary with the units of the model properties to initialize 
            correct conversion factors.

        """
        raise NotImplementedError

    def get_trainable_parameters(
        self,
        no_weight_decay: Optional[bool] = True,
    ) -> Dict[str, List]:
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
            else:
                trainable_parameters['default'].append(parameter)

        return trainable_parameters

    # @torch.compile # Not supporting backwards propagation with torch.float64
    # @torch.jit.export  # No effect, as 'forward' already is
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
                    Atom j pair index pointer from image atom to respective
                    primary atom index in a supercluster

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """
        raise NotImplementedError

    def calculate(
        self,
        atoms: ase.Atoms,
        charge: Optional[float] = 0.0,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of PhysNet Calculator model from ASE Atoms object.

        Parameters
        ----------
        atoms: ase.Atoms
            ASE Atoms object to calculate properties
        charge: float, optional, default 0.0
            Total system charge

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Initialize atoms batch
        atoms_batch = {}

        # Number of atoms
        Natoms = len(atoms)
        atoms_batch['atoms_number'] = torch.tensor(
            [Natoms], dtype=torch.int64)

        # Atomic number
        atoms_batch['atomic_numbers'] = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.int64)

        # Atom positions
        atoms_batch['positions'] = torch.zeros(
            [Natoms, 3], dtype=torch.float64)

        # Atom periodic boundary conditions
        atoms_batch['pbc'] = torch.tensor(
            atoms.get_pbc(), dtype=torch.bool)

        # Atom cell information
        atoms_batch['cell'] = torch.tensor(
            atoms.get_cell()[:], dtype=torch.float64)

        # Atom segment indices, just one atom segment allowed
        atoms_batch['sys_i'] = torch.zeros(
            Natoms, dtype=torch.int64)

        # Total atomic system charge
        atoms_batch['charge'] = torch.tensor(
            [charge], dtype=torch.float64)

        neighbor_list = utils.TorchNeighborListRangeSeparated(
            self.model_cutoff,
            self.device,
            self.dtype)
        atoms_batch = neighbor_list(atoms_batch)

        return self.forward(atoms_batch)
