import numpy as np
import torch

# ======================================
# General Model Type Settings
# ======================================

# Default calculator model
_default_calculator_model = 'PhysNet'

# Available input model of respective 'model_type'
_available_input_model = {
    'PhysNet':                      'PhysNetRBF',
    'PhysNet_original':             'PhysNetRBF_original',
    'PaiNN':                        'PaiNNRBF',
    }

# Available graph model of respective 'model_type'
_available_graph_model = {
    'PhysNet':                      'PhysNetMP',
    'PhysNet_original':             'PhysNetMP',
    'PaiNN':                        'PaiNNMP',
    }

# Available output model of respective 'model_type'
_available_output_model = {
    'PhysNet':                      'PhysNetOut',
    'PhysNet_original':             'PhysNetOut',
    'PaiNN':                        'PaiNNOut',
    }

# ======================================
# Default Input
# ======================================

# Default arguments for input variables
_default_args = {
    'config':                       {},
    'config_file':                  'config.json',
    # Model
    'model_type':                   None,
    'model_directory':              None,
    'model_restart':                False,
    'model_calculator':             None,
    'model_num_threads':            None,
    'model_path':                   'best_model',
    'model_properties':             ['energy', 'forces'],
    'model_unit_properties':        None,
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
    'input_type':                   None,
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
    'graph_type':                   None,
    'graph_n_blocks':               5,
    'graph_n_residual_interaction': 3,
    'graph_n_residual_atomic':      2,
    'graph_activation_fn':          None,
    'graph_stability_constant':     1.e-8,
    # Output module
    'output_calculator':            None,
    'output_type':                  None,
    'output_n_residual':            1,
    'output_properties':            ['energy', 'forces'],
    'output_unit_properties':       {'energy': 'eV',
                                     'forces': 'eV/Ang',
                                     'atomic_charges': 'e'},
    'output_activation_fn':         None,
    # Trainer
    'trainer_restart':              False,
    'trainer_max_epochs':           10_000,
    'trainer_loss_fn_properties':   {'energy': 'MAE', 'forces': 'MAE'},
    'trainer_loss_weight':          {'energy': 1., 'forces': 50.},
    'trainer_properties_train':     [],
    'trainer_properties_metrics':   {'else': 'MSE'},
    'trainer_properties_weights':   {'energy': 1., 'forces': 50., 'else': 1.},
    'trainer_optimizer':            'AMSgrad',
    'trainer_optimizer_args':       {'lr': 0.001, 'weight_decay': 1.e-5},
    'trainer_scheduler':            'ExponentialLR',
    'trainer_scheduler_args':       {'gamma': 0.999},
    'trainer_ema':                  True,
    'trainer_ema_decay':            0.99,
    'trainer_max_gradient_norm':    1000.0,
    'trainer_save_interval':        5,
    'trainer_validation_interval':  5,
    'trainer_dropout_rate':         0.0,
    'trainer_evaluate_testset':     True,
    'trainer_max_checkpoints':      1,
    'trainer_store_neighbor_list':  False,
    # Tester
    'test_datasets':                ['test'],
    'tester_properties':            None,
    'test_store_neighbor_list':     False,
    # Sample Re-calculator
    'recalc_interface':             'ase',
    'recalc_calculator':            'XTB',
    'recalc_calculator_args':       {'charge': 0},
    'recalc_properties':            ['energy', 'forces', 'dipole'],
    'recalc_source_data_file':      None,
    'recalc_target_data_file':      None,
    'recalc_directory':             '',
}
