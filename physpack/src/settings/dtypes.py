
from .. import utils

#======================================
# Input data types
#======================================

# Expected data types of input variables
_dtypes_args = {
    # Model
    'model_directory':              [utils.is_string],
    'model_calculator':             [utils.is_object],
    'model_type':                   [utils.is_string],
    'model_properties':             [utils.is_string_array],
    'model_save_top_k':             [utils.is_integer],
    'model_interaction_cutoff':     [utils.is_numeric],
    'model_cutoff_width':           [utils.is_numeric],
    'model_repulsion':              [utils.is_bool],
    'model_dispersion':             [utils.is_bool],
    'model_electrostatic':          [utils.is_bool],
    'model_activation_fn':          [utils.is_string, utils.is_object],
    'model_rate':                   [utils.is_numeric],
    'model_device':                 [utils.is_string],
    # Input module
    'input_model':                  [utils.is_object],
    'input_type':                   [utils.is_string],
    'input_cutoff_descriptor':      [utils.is_numeric],
    'input_cutoff_fn':              [utils.is_string, utils.is_object],
    # Representation module
    'graph_model':                  [utils.is_object],
    'graph_type':                   [utils.is_string],
    'graph_n_atombasis':            [utils.is_integer],
    'graph_n_interaction':          [utils.is_integer],
    'graph_activation_fn':          [utils.is_string, utils.is_object],
    # Output module
    'output_model':                 [utils.is_object],
    'output_type':                  [utils.is_string],
    'output_n_residual':            [utils.is_integer],
    'output_activation_fn':         [utils.is_string, utils.is_object],
    'output_properties':            [utils.is_string_array],
    'output_unit_properties':       [utils.is_dictionary],
    # DataContainer & DataSet
    'data_container':               [],
    'data_file':                    [utils.is_string],
    'data_source':                  [utils.is_string, utils.is_string_array],
    'data_format':                  [utils.is_string, utils.is_string_array],
    'data_unit_positions':          [utils.is_string],
    'data_load_properties':         [utils.is_array_like],
    'data_unit_properties':         [utils.is_dictionary],
    'data_alt_property_labels':     [utils.is_dictionary],
    'data_num_train':               [utils.is_numeric],
    'data_num_valid':               [utils.is_numeric],
    'data_num_test':                [utils.is_numeric],
    'data_train_batch_size':        [utils.is_integer],
    'data_val_batch_size':          [utils.is_integer],
    'data_test_batch_size':         [utils.is_integer],
    'data_num_workers':             [utils.is_integer],
    'data_data_overwrite':          [utils.is_bool],
    'data_seed':                    [utils.is_numeric],
    'data_workdir':                 [utils.is_string],
    # DataSubSet
    'dataset':                      [utils.is_object],
    'subset_idx':                   [utils.is_integer_array],
    # Trainer
    'trainer_max_epochs':           [utils.is_integer],
    'trainer_properties_train':     [utils.is_string_array],
    'trainer_properties_metrics':   [utils.is_dictionary],
    'trainer_properties_weights':   [utils.is_dictionary],
    'trainer_optimizer':            [utils.is_string, utils.is_object],
    'trainer_optimizer_args':       [utils.is_dictionary],
    'trainer_scheduler':            [utils.is_string, utils.is_object],
    'trainer_scheduler_args':       [utils.is_dictionary],
    'trainer_ema':                  [utils.is_bool],
    'trainer_ema_decay':            [utils.is_numeric],
    'trainer_max_gradient_norm':    [utils.is_numeric],
    'trainer_save_interval':        [utils.is_integer],
    'trainer_validation_interval':  [utils.is_integer],
    'trainer_dropout_rate':         [utils.is_numeric],
    # Sampler
    'sample_directory':             [utils.is_string],
    'sample_systems':               [utils.is_string,
                                     utils.is_string_array,
                                     utils.is_object],
    'sample_systems_format':        [utils.is_string, utils.is_string_array],
    'sample_calculator':            [utils.is_string, utils.is_object],
    'sample_calculator_args':       [utils.is_dictionary],
    'sample_systems_optimize':      [utils.is_bool],
    'sample_systems_optimize_fmax': [utils.is_numeric],
    'sample_ref_calculator':        [utils.is_string, utils.is_object],
    'sample_ref_calculator_args':   [utils.is_dictionary],
    }
