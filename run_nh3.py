from asparagus import Asparagus

# Sampling
if True:
    
    from asparagus import NormalModeScanner
    
    sampler = NormalModeScanner(
        config='nh3_config.json',
        sample_directory='model_nh3/sampling',
        sample_data_file='data/nms_nh3.db',
        #sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems=['data/nh3_c3v.xyz'],
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.01,
        nms_energy_limits=1.00,
        nms_number_of_coupling=2,
        )
    sampler.run()

# Train
if True:
    
    model = Asparagus(
        config='nh3_config.json',
        data_file='data/nms_nh3.db',
        model_directory='model_nh3',
        model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        model_electrostatic=True,
        model_interaction_cutoff=12.0,
        input_cutoff_descriptor=6.0,
        model_cutoff_width=2.0,
        trainer_properties_weights={
            'energy': 1.,
            'forces': 50.,
            'dipole': 20.
            },
        trainer_optimizer_args={'lr': 0.001, 'weight_decay': 0.e-5},
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))
