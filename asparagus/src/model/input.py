
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import torch
#import pytorch_lightning as pl

from .. import settings
from .. import utils
from .. import layers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_input_model', 'Input_PhysNetRBF']

# ======================================
#  Input Models
# ======================================

class Input_PhysNetRBF(torch.nn.Module):
    """
    PhysNet input model class
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        input_n_atombasis: Optional[int] = None,
        input_n_radialbasis: Optional[int] = None,
        input_cutoff_descriptor: Optional[float] = None,
        input_cutoff_fn: Optional[Union[str, object]] = None,
        input_rbf_center_start: Optional[float] = None,
        input_rbf_center_end: Optional[float] = None,
        input_rbf_trainable: Optional[bool] = None,
        input_n_maxatom: Optional[int] = None,
        input_atom_features_range: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize PhysNet input model.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        input_n_atombasis: int
            Number of atom property features (atomic feature vector lengths)
        input_n_radialbasis: int
            Number of structural fingerprint features
        input_cutoff_descriptor: float
            Upper cutoff atom distance for including atom environment
        input_cutoff_fn: class object
            Cutoff function class for weighting atom environment
        input_rbf_center_start: float
            Initial shortest center of radial basis functions
        input_rbf_center_end: float
            Initial largest center of radial basis functions
        input_rbf_trainable: bool
            If True, radial basis function parameter such as center and width
            are optimized during training. If False, radial basis function 
            parameter are fixed.
        input_n_maxatom: int
            Highest atom order number to initialize isolated atom feature 
            vector library
        input_atom_features_range: float
            Range for uniform distribution of initial random atom feature 
            vector library
        **kwargs: dict, optional
            Additional arguments for parameter initialization 

        Returns
        -------
        callable object
            PhysNet RBF input model object
        """

        super(Input_PhysNetRBF, self).__init__()
        
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

        # Input class type
        self.input_type = 'RBF_PhysNet'
        
        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device
        
        # Initialize cutoff function
        self.input_cutoff_fn = layers.get_cutoff_fn(self.input_cutoff_fn)(
            self.input_cutoff_descriptor)
        
        # Additional arguments (set as class variable next by setattr(self,...))
        
        # Initialize Atomic feature vectors up to Plutonium (Pu95)
        self.atom_features = torch.nn.Parameter(
            torch.empty(
                self.input_n_maxatom + 1, self.input_n_atombasis, 
                dtype=self.dtype,
                device=self.device
                ).uniform_(
                    -self.input_atom_features_range, 
                    self.input_atom_features_range)
            )
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_cutoff_descriptor
        
        # Radial distribution function
        radial_fn = layers.get_radial_fn(self.input_type)
        self.input_descriptor_fn = radial_fn(
            self.input_n_radialbasis, self.input_cutoff_fn, 
            self.input_rbf_center_start, self.input_rbf_center_end,
            self.input_rbf_trainable, device=self.device, dtype=self.dtype)


    def forward(
        self, 
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        pbc_offset: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        
        # Collect atom feature vectors
        features = self.atom_features[atomic_numbers]

        # Compute pair distances
        if pbc_offset is None:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i],
                dim=-1
                )
        else:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i] + pbc_offset,
                dim=-1
                )

        # Compute radial fingerprint
        rbfs = self.input_descriptor_fn(distances)
        
        return features, rbfs, distances
        
        
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'input_type': self.input_type,
            'input_n_atombasis': self.input_n_atombasis,
            'input_n_radialbasis': self.input_n_radialbasis,
            'input_cutoff_descriptor': self.input_cutoff_descriptor,
            }
    

class Input_PhysNetRBF_original(torch.nn.Module):
    """
    Original PhysNet input model class
    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        input_n_atombasis: Optional[int] = None,
        input_n_radialbasis: Optional[int] = None,
        input_cutoff_descriptor: Optional[float] = None,
        input_cutoff_fn: Optional[Union[str, object]] = None,
        input_rbf_center_start: Optional[float] = None,
        input_rbf_center_end: Optional[float] = None,
        input_rbf_trainable: Optional[bool] = None,
        input_n_maxatom: Optional[int] = None,
        input_atom_features_range: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize original PhysNet input model.

        Parameters
        ----------

        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        input_n_atombasis: int
            Number of atom property features (atomic feature vector lengths)
        input_n_radialbasis: int
            Number of structural fingerprint features
        input_cutoff_descriptor: float
            Upper cutoff atom distance for including atom environment
        input_cutoff_fn: class object
            Cutoff function class for weighting atom environment
        input_rbf_center_start: float
            Initial shortest center of radial basis functions
        input_rbf_center_end: float
            Initial largest center of radial basis functions
        input_rbf_trainable: bool
            If True, radial basis function parameter such as center and width
            are optimized during training. If False, radial basis function 
            parameter are fixed.
        input_n_maxatom: int
            Highest atom order number to initialize isolated atom feature 
            vector library
        input_atom_features_range: float
            Range for uniform distribution of initial random atom feature 
            vector library
        **kwargs: dict, optional
            Additional arguments for parameter initialization 

        Returns
        -------
        callable object
            PhysNet RBF input model object
        """

        super(Input_PhysNetRBF, self).__init__()
        
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

        # Input class type
        self.input_type = 'RBF_PhysNet_original'
        
        # Assign global arguments
        self.dtype = settings._global_dtype
        self.device = settings._global_device
        
        # Initialize cutoff function
        self.input_cutoff_fn = layers.get_cutoff_fn(self.input_cutoff_fn)(
            self.input_cutoff_descriptor)
        
        # Additional arguments (set as class variable next by setattr(self,...))
        
        # Initialize Atomic feature vectors up to Plutonium (Pu95)
        self.atom_features = torch.nn.Parameter(
            torch.empty(
                self.input_n_maxatom + 1, self.input_n_atombasis, 
                dtype=self.dtype,
                device=self.device
                ).uniform_(
                    -self.input_atom_features_range, 
                    self.input_atom_features_range)
            )
        
        # Get upper RBF center range
        if self.input_rbf_center_end is None:
            self.input_rbf_center_end = self.input_cutoff_descriptor
        
        # Radial distribution function
        radial_fn = layers.get_radial_fn(self.input_type)
        self.input_descriptor_fn = radial_fn(
            self.input_n_radialbasis, self.input_cutoff_fn, 
            self.input_rbf_center_start, self.input_rbf_center_end,
            self.input_rbf_trainable, device=self.device, dtype=self.dtype)


    def forward(
        self, 
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        pbc_offset: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        
        # Collect atom feature vectors
        features = self.atom_features[atomic_numbers]

        # Compute pair distances
        if pbc_offset is None:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i],
                dim=-1
                )
        else:
            distances = torch.norm(
                positions[idx_j] - positions[idx_i] + pbc_offset,
                dim=-1
                )

        # Compute radial fingerprint
        rbfs = self.input_descriptor_fn(distances)
        
        return features, rbfs, distances
        
        
    def get_info(self) -> Dict[str, Any]:
        """
        Return class information
        """
        
        return {
            'input_type': self.input_type,
            'input_n_atombasis': self.input_n_atombasis,
            'input_n_radialbasis': self.input_n_radialbasis,
            'input_cutoff_descriptor': self.input_cutoff_descriptor,
            }


# ======================================
#  Input Model Assignment
# ======================================

input_model_available = {
    'PhysNetRBF'.lower(): Input_PhysNetRBF,
    'PhysNetRBF_original'.lower(): Input_PhysNetRBF_original,
    }

def get_input_model(
    config: Optional[Union[str, dict, object]] = None,
    input_type: Optional[str] = None,
    **kwargs,
) -> Callable:
    """
    Input module selection

    Parameters
    ----------

    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    input_type: str
        Input model representation of the atomistic structural information
        e.g. 'PhysNetRBF'
    **kwargs: dict, optional
        Additional arguments for parameter initialization

    Returns
    -------
        callable object
            Input model object to encode atomistic structural information
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

    # Check input model type
    if config.get('input_type') is None:
        model_type = config.get('model_type')
        if settings._available_input_model.get(model_type) is None:
            raise SyntaxError(
                "No input model type could assigned from defined model "
                + f"type '{model_type:s}'!")
        config['input_type'] = settings._available_input_model.get(model_type)
    input_type = config['input_type']

    # Input model type assignment
    if (
        input_type.lower() in 
        [key.lower() for key in input_model_available.keys()]
    ):
        return input_model_available[input_type.lower()](
            config,
            **kwargs)
    else:
        raise ValueError(
            f"Input model type input '{input_type:s}' is not valid!" +
            "Choose from:\n" + str(input_model_available.keys()))
    
    return
