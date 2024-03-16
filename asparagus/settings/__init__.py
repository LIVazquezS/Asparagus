"""
Basic settings for the Asparagus package.

Definition of global variables and functions to set them, additionally it 
handles the configuration file.

"""

from .default import (
    _default_args, _default_calculator_model
)

from .dtypes import (
    _dtypes_args, _dtype_library
)

from .properties import (
    _valid_cutoff_fn, _default_property_labels, _valid_properties,
    _alt_property_labels, _default_units, _default_output_block_options
)

from .globl import (
    _global_mode, set_global_mode,
    _global_config_file, set_global_config_file,
    _global_dtype, set_global_dtype,
    _global_device, set_global_device,
    _global_rate, set_global_rate
)

from .config import (
    Configuration, get_config
)
