
from .. import utils

#======================================
# Input data types
#======================================

# Expected data types of input variables
_dtypes_args = {
    # Model
    'model_directory':              [utils.is_string, utils.is_None],
    'model_num_threads':            [utils.is_integer, utils.is_None],
    'model_calculator':             [utils.is_callable],
    'model_type':                   [utils.is_string, utils.is_None],
    'model_properties':             [utils.is_string_array],
    'model_unit_properties':        [utils.is_dictionary, utils.is_None],
    'model_save_top_k':             [utils.is_integer],
    'model_interaction_cutoff':     [utils.is_numeric],
    'model_cutoff_width':           [utils.is_numeric],
    'model_repulsion':              [utils.is_bool],
    'model_dispersion':             [utils.is_bool],
    'model_electrostatic':          [utils.is_bool],
    'model_activation_fn':          [utils.is_string, utils.is_callable],
    'model_rate':                   [utils.is_numeric],
    'model_device':                 [utils.is_string],
    # Input module
    'input_model':                  [utils.is_callable],
    'input_type':                   [utils.is_string, utils.is_None],
    'input_cutoff_descriptor':      [utils.is_numeric],
    'input_cutoff_fn':              [utils.is_string, utils.is_callable],
    # Representation module
    'graph_model':                  [utils.is_callable],
    'graph_type':                   [utils.is_string, utils.is_None],
    'graph_n_atombasis':            [utils.is_integer],
    'graph_n_interaction':          [utils.is_integer],
    'graph_activation_fn':          [utils.is_string, utils.is_callable],
    'graph_stability_constant':     [utils.is_numeric],
    # Output module
    'output_model':                 [utils.is_callable],
    'output_type':                  [utils.is_string, utils.is_None],
    'output_n_residual':            [utils.is_integer],
    'output_activation_fn':         [utils.is_string, utils.is_callable],
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
    'dataset':                      [utils.is_callable],
    'subset_idx':                   [utils.is_integer_array],
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
    # Sampler
    'sample_directory':             [utils.is_string, utils.is_None],
    'sample_data_file':             [utils.is_string, utils.is_None],
    'sample_systems':               [utils.is_string,
                                     utils.is_string_array,
                                     utils.is_ase_atoms,
                                     utils.is_ase_atoms_array],
    'sample_systems_format':        [utils.is_string, utils.is_string_array],
    'sample_calculator':            [utils.is_string, utils.is_object],
    'sample_calculator_args':       [utils.is_dictionary],
    'sample_properties':            [utils.is_string, utils.is_string_array],
    'sample_systems_optimize':      [utils.is_bool, utils.is_boolean_array],
    'sample_systems_optimize_fmax': [utils.is_numeric],
    'sample_data_overwrite':        [utils.is_bool],
    'sample_tag':                   [utils.is_string],
    'nms_harmonic_energy_step':     [utils.is_numeric],
    'nms_energy_limits':            [utils.is_numeric, utils.is_numeric_array],
    'nms_number_of_coupling':       [utils.is_numeric],
    'nms_limit_of_steps':           [utils.is_numeric],
    'nms_limit_com_shift':          [utils.is_numeric],
    'md_temperature':               [utils.is_numeric],
    'md_time_step':                 [utils.is_numeric],
    'md_simulation_time':           [utils.is_numeric],
    'md_save_interval':             [utils.is_integer],
    'md_langevin_friction':         [utils.is_numeric],
    'md_equilibration_time':        [utils.is_numeric],
    'md_initial_velocities':        [utils.is_bool],
    'md_initial_temperature':       [utils.is_numeric],
    'meta_cv':                      [utils.is_array_like],
    'meta_gaussian_height':         [utils.is_numeric],
    'meta_gaussian_width':          [utils.is_numeric, utils.is_numeric_array],
    'meta_gaussian_interval':       [utils.is_integer],
    'meta_hookean':                 [utils.is_array_like],
    'meta_hookean_force_constant':  [utils.is_numeric],
    'meta_temperature':             [utils.is_numeric],
    'meta_time_step':               [utils.is_numeric],
    'meta_simulation_time':         [utils.is_numeric],
    'meta_save_interval':           [utils.is_integer],
    'meta_langevin_friction':       [utils.is_numeric],
    'meta_initial_velocities':      [utils.is_bool],
    'meta_initial_temperature':     [utils.is_numeric],
    # Sample Re-calculator
    'recalc_interface':             [utils.is_string],
    'recalc_calculator':            [utils.is_string],
    'recalc_calculator_args':       [utils.is_dictionary],
    'recalc_properties':            [utils.is_string, utils.is_string_array],
    'recalc_source_data_file':      [utils.is_string, utils.is_string_array],
    'recalc_target_data_file':      [utils.is_string, utils.is_string_array],
    'recalc_directory':             [utils.is_string],
    }
