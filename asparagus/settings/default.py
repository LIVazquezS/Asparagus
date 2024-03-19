import numpy as np
import torch

# ======================================
# General Model Type Settings
# ======================================

# Default calculator model
_default_calculator_model = 'PhysNet'

# ======================================
# Default Input
# ======================================

# Default arguments for input variables
_default_args = {
    'config':                       {},
    'config_file':                  'config.json',
    'device':                       'cpu',
    'dtype':                        torch.float64,
    # Model
    'model_calculator':             None,
    'model_type':                   None,
    'model_num_threads':            None,
    'model_restart':                False,
    'model_path':                   'best_model',
    'model_device':                 'cpu',
    'model_dtype':                  torch.float64,
    'model_seed':                   np.random.randint(1E6),
    # Input module
    'input_calculator':             None,
    'input_type':                   None,
    # Graph module
    'graph_calculator':             None,
    'graph_type':                   None,
    'graph_stability_constant':     1.e-8,
    # Output module
    'output_calculator':            None,
    'output_type':                  None,
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
