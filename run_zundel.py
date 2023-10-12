from asparagus import Asparagus

# Sampling
if False:
    
    from asparagus import MetaSampler
    
    sampler = MetaSampler(
        config='zundel_config.json',
        sample_directory='model_zundel/sampling',
        sample_systems=['data/zundel_h5o2.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 1,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 4 end',
            'directory': 'model_zundel/sampling'},
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.01,
        meta_cv=[['-', 0, 1, 0, 4], [1, 4]],
        meta_hookean=[[1, 4, 8.0]],
        meta_gaussian_height=0.05,
        meta_gaussian_widths=0.2,
        meta_gaussian_interval=10,
        meta_time_step=1.0,
        meta_simulation_time=10000.0,
        meta_save_interval=20,
        meta_temperature=300,
        meta_langevin_friction=1.0,
        meta_initial_velocities=True,
        meta_initial_temperature=300.,
        )
    sampler.run()

# Collect Data
if False:
    
    from asparagus import DataContainer
    
    data = DataContainer(
        config='zundel_config.json',
        data_file='data/zundel.db',
        data_source=[
            'model_zundel/sampling/1_meta.db',
            ],
        data_format=[
            'db',
            ],
        data_load_properties=[
            'energy', 'forces', 'charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'forces':   'eV/Ang',
            'charge':   'e',
            'dipole':   'e*Ang'},
        data_overwrite=True)

# Train
if False:
    
    model = Asparagus(
        config='zundel_config.json',
        data_file='data/zundel.db',
        model_directory='model_zundel',
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

if True:

    import ase
    from ase.optimize import BFGS
    from ase.visualize import view
    from ase.vibrations import Vibrations
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.io.trajectory import Trajectory
    import ase.units as units
    
    model = Asparagus(
        config="model_zundel/config.json",
        model_directory="model_zundel"
        )
    
    calc = model.get_ase_calculator()
    zundel = ase.io.read("data/zundel_h7o3.xyz")
    zundel.calc = calc
    
    dyn = BFGS(zundel)
    dyn.run(fmax=0.01)
    
    view(zundel)
    
    vib = Vibrations(zundel)
    vib.clean()
    vib.run()
    vib.summary()
    vib.write_mode()
    
    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(zundel, temperature_K=300)

    # We want to run MD with constant energy using the VelocityVerlet algorithm.
    dyn = VelocityVerlet(zundel, 0.5 * units.fs)
         
    traj = Trajectory('zundel.traj', 'w', zundel)
    dyn.attach(traj.write, interval=1)
    dyn.run(2000)
