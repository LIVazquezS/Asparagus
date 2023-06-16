import numpy as np
import torch

# ======================================
# Default Models
# ======================================

# Default calculator model
_default_calculator_model = 'PhysNet'

# Default input model of respective 'model_type'
_default_input_model = {
    'PhysNet':                      'PhysNetRBF',
    }

# Default graph model of respective 'model_type'
_default_graph_model = {
    'PhysNet':                      'PhysNetMP',
    }

# Default output model of respective 'model_type'
_default_output_model = {
    'PhysNet':                      'PhysNetOut',
    }

# ======================================
# Default Input
# ======================================

# Default arguments for input variables
_default_args = {
    'config':                       {},
    'config_file':                  'config.json',
    # Model
    'model_directory':              None,
    'model_restart':                False,
    'model_calculator':             None,
    'model_type':                   _default_calculator_model,
    'model_path':                   'best_model',
    'model_properties':             ['energy', 'forces'],
    'model_save_top_k':             1,
    'model_interaction_cutoff':     8.0,
    'model_cutoff_width':           2.0,
    'model_repulsion':              False,
    'model_dispersion':             True,
    'model_electrostatic':          True,
    'model_activation_fn':          'shifted_softplus',
    'model_dispersion_trainable':   True,
    'model_device':                 'cpu',
    'model_dtype':                  torch.float64,
    'model_seed':                   np.random.randint(1E6),
    # Input module
    'input_calculator':             None,
    'input_type':                   _default_input_model.get(
                                        _default_calculator_model),
    'input_n_atombasis':            128,
    'input_n_radialbasis':          64,
    'input_cutoff_descriptor':      8.0,
    'input_cutoff_fn':              'Poly6',
    'input_rbf_center_start':       1.0,
    'input_rbf_center_end':         None,
    'input_rbf_trainable':          True,
    'input_n_maxatom':              94,
    'input_atom_features_range':    np.sqrt(3),
    # Graph module
    'graph_calculator':             None,
    'graph_type':                   _default_graph_model.get(
                                        _default_calculator_model),
    'graph_n_blocks':               5,
    'graph_n_residual_interaction': 3,
    'graph_n_residual_atomic':      2,
    'graph_activation_fn':          None,
    # Output module
    'output_calculator':            None,
    'output_type':                  _default_output_model.get(
                                        _default_calculator_model),
    'output_n_residual':            1,
    'output_properties':            ['energy', 'forces'],
    'output_unit_properties':       {'energy': 'eV',
                                     'forces': 'eV/Ang',
                                     'atomic_charges': 'e'},
    'output_activation_fn':         None,
    # DataContainer & DataSet
    'data_container':               None,
    'data_file':                    'data.db',
    'data_source':                  [],
    'data_format':                  [],
    'data_unit_positions':          'Ang',
    'data_load_properties':         ['energy', 'forces', 'charge', 'dipole'],
    'data_unit_properties':         {'energy': 'eV',
                                     'forces': 'eV/Ang',
                                     'charge': 'e',
                                     'dipole': 'eAng'},
    'data_alt_property_labels':     {},
    'data_num_train':               0.8,
    'data_num_valid':               0.1,
    'data_num_test':                None,
    'data_train_batch_size':        128,
    'data_valid_batch_size':        128,
    'data_test_batch_size':         128,
    'data_num_workers':             1,
    'data_workdir':                 '.',
    'data_overwrite':               False,
    'data_seed':                    np.random.randint(1E6),
    # Trainer
    'trainer_max_epochs':           10_000,
    'trainer_loss_fn_properties':   {'energy': 'MAE', 'forces': 'MAE'},
    'trainer_loss_weight':          {'energy': 1., 'forces': 50.},
    'trainer_properties_train':     [],
    'trainer_properties_metrics':   {'else': 'MSE'},
    'trainer_properties_weights':   {'energy': 1., 'forces': 50., 'else': 1.},
    'trainer_optimizer':            'AMSgrad',
    'trainer_optimizer_args':       {'lr': 0.001, 'weight_decay': 1.e-5},
    'trainer_scheduler':            'ExponentialLR',
    'trainer_scheduler_args':       {'gamma': 0.999, 'decay_steps': 1000},
    'trainer_ema':                  True,
    'trainer_ema_decay':            0.999,
    'trainer_max_gradient_norm':    1000.0,
    'trainer_save_interval':        5,
    'trainer_validation_interval':  5,
    'trainer_dropout_rate':         0.0,
    # Sampler
    'sample_directory':             '',
    'sample_data_file':             'sample.db',
    'sample_systems':               None,
    'sample_systems_format':        None,
    'sample_calculator':            'XTB',
    'sample_calculator_args':       {},
    'sample_properties':            ['energy', 'forces', 'dipole'],
    'sample_systems_optimize':      False,
    'sample_systems_optimize_fmax': 0.001,
    'nms_harmonic_energy_step':     0.05,
    'nms_energy_limits':            1.0,
    'nms_limit_of_coupling':        2,
    'nms_limit_of_steps':           10,
    'md_temperature':               300.,
    'md_time_step':                 1.,
    'md_simulation_time':           1.E5,
    'md_save_interval':             100,
    'md_langevin_friction':         1.E-2,
    'md_equilibration_time':        None,
    'md_initial_velocities':        False,
    'md_initial_temperature':       300.,
    'meta_cv':                      [],
    'meta_gaussian_height':         0.05,
    'meta_gaussian_widths':         0.1,
    'meta_gaussian_interval':       10,
    'meta_hookean':                 [],
    'meta_hookean_force_constant':  5.0,
    'meta_temperature':             300.,
    'meta_time_step':               1.,
    'meta_simulation_time':         1.E5,
    'meta_save_interval':           100,
    'meta_langevin_friction':       1.E-1,
    'meta_initial_velocities':      False,
    'meta_initial_temperature':     300.,
}
