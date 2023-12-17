from asparagus import Asparagus

# Sampling
if False:

    # System generation
    from ase.build import add_adsorbate, fcc100
    from ase.constraints import FixAtoms, FixedPlane
    from ase.optimize import QuasiNewton
    from ase import io

    # 2x2-Al(001) surface with 3 layers and an
    # Au atom adsorbed in a hollow site:
    slab = fcc100('Al', size=(2, 2, 3))
    add_adsorbate(slab, 'Au', 1.45, 'hollow')
    slab.center(axis=2, vacuum=4.0)

    # Fix second and third layers:
    mask = [atom.tag > 1 for atom in slab]
    fixlayers = FixAtoms(mask=mask)

    # Load optimized slab
    slab = io.Trajectory("model_diffusion/sampling/1_bfgs.traj")[-1]

    # Set constraint
    slab.set_constraint([fixlayers])

    indices = range(len(slab))
    for constraint in slab.constraints:
        if isinstance(constraint, FixAtoms):
            indices = [idx for idx in indices if idx not in constraint.index]

    # Get GPAW calculator
    from gpaw import GPAW, PW
    calc = GPAW(
        mode=PW(200),
        kpts=(2, 2, 1),
        xc='PBE',
        symmetry='off')

    # First: Normal Mode Scan
    from asparagus import NormalModeScanner
    sampler = NormalModeScanner(
        config='difftest_config.json',
        sample_directory='model_difftest/sampling',
        sample_data_file='model_difftest/nms_difftest.db',
        sample_systems=[slab],
        sample_calculator=calc,
        sample_systems_optimize=False,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.05,
        nms_energy_limits=1.00,
        nms_number_of_coupling=1,
        nms_limit_com_shift=1.0,
        nms_limit_of_steps=25,
        )
    sampler.run()

if True:

    model = Asparagus(
        config='difftest_config.json',
        data_file='model_difftest/nms_difftest.db',
        model_type='PhysNet',
        model_directory="model_difftest",
        model_num_threads=4,
        model_interaction_cutoff=12.0,
        input_cutoff_descriptor=8.0,
        model_properties=['energy', 'forces', 'atomic_charges'],
        data_num_train=100,
        data_num_valid=10,
        data_num_test=7,
        data_train_batch_size=16,
        trainer_max_epochs=1_000,
        trainer_properties_weights={
            'energy': 1., 'forces': 50., 'dipole': 50.},
        trainer_validation_interval=1,
        trainer_store_neighbor_list=True,
        )
    # Start training
    model.train()
