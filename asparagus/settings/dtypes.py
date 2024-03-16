import numpy as np

import torch

from .. import utils

#======================================
# Input data types
#======================================

# Expected data types of input variables
_dtypes_args = {
    # Model
    'model_calculator':             [utils.is_callable],
    'model_type':                   [utils.is_string, utils.is_None],
    'model_directory':              [utils.is_string, utils.is_None],
    'model_path':                   [utils.is_string],
    'model_num_threads':            [utils.is_integer, utils.is_None],
    'model_save_top_k':             [utils.is_integer],
    'model_device':                 [utils.is_string],
    'model_seed':                   [utils.is_integer],
    # Input module
    'input_model':                  [utils.is_callable],
    'input_type':                   [utils.is_string, utils.is_None],
    # Representation module
    'graph_model':                  [utils.is_callable],
    'graph_type':                   [utils.is_string, utils.is_None],
    'graph_stability_constant':     [utils.is_numeric],
    # Output module
    'output_model':                 [utils.is_callable],
    'output_type':                  [utils.is_string, utils.is_None],
    # Trainer
    'trainer_restart':              [utils.is_bool],
    'trainer_max_epochs':           [utils.is_integer],
    'trainer_properties_train':     [utils.is_string_array],
    'trainer_properties_metrics':   [utils.is_dictionary],
    'trainer_properties_weights':   [utils.is_dictionary],
    'trainer_optimizer':            [utils.is_string, utils.is_callable],
    'trainer_optimizer_args':       [utils.is_dictionary],
    'trainer_scheduler':            [utils.is_string, utils.is_callable],
    'trainer_scheduler_args':       [utils.is_dictionary],
    'trainer_ema':                  [utils.is_bool],
    'trainer_ema_decay':            [utils.is_numeric],
    'trainer_max_gradient_norm':    [utils.is_numeric],
    'trainer_save_interval':        [utils.is_integer],
    'trainer_validation_interval':  [utils.is_integer],
    'trainer_dropout_rate':         [utils.is_numeric],
    'trainer_evaluate_testset':     [utils.is_bool],
    'trainer_max_checkpoints':      [utils.is_integer],
    'trainer_store_neighbor_list':  [utils.is_bool],
    # Tester
    'test_datasets':                [utils.is_string, utils.is_string_array],
    'tester_properties':            [utils.is_string,
                                     utils.is_string_array,
                                     utils.is_None],
    'test_store_neighbor_list':     [utils.is_bool],
    # Sample Re-calculator
    'recalc_interface':             [utils.is_string],
    'recalc_calculator':            [utils.is_string],
    'recalc_calculator_args':       [utils.is_dictionary],
    'recalc_properties':            [utils.is_string, utils.is_string_array],
    'recalc_source_data_file':      [utils.is_string, utils.is_string_array],
    'recalc_target_data_file':      [utils.is_string, utils.is_string_array],
    'recalc_directory':             [utils.is_string],
    }

#======================================
# Python data type library
#======================================

_dtype_library = {
    'float': float, 
    'np.float16': np.float16, 
    'np.float32': np.float32, 
    'np.float64': np.float64,
    'torch.float': torch.float,
    'torch.float16': torch.float16,
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    }
