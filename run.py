
from asparagus import DataContainer

from asparagus import MDSampler, NormalModeScanner, MetaSampler, ReCalculator

from asparagus import Asparagus
from asparagus.src.layers import PC_shielded_electrostatics
import torch
import matplotlib.pyplot as plt
# Test electrostatic

#charges = torch.Tensor([-0.1, 0.1])
#distances = torch.arange(0.2, 14.0, .2)
#idx_i = torch.zeros_like(distances).to(torch.int64)
#idx_j = torch.ones_like(distances).to(torch.int64)


#elec1 = PC_shielded_electrostatics(
    #True,
    #6.0,
    #12.0,
    #{'energy': 'eV', 'positions': 'Ang', 'charge': 'e'},
    #)

#elec2 = PC_shielded_electrostatics(
    #False,
    #12.0,
    #12.0,
    #{'energy': 'eV', 'positions': 'Ang', 'charge': 'e'},
    #)

#E1, Eo1 = elec1(charges, distances, idx_i, idx_j)
#E2, Eo2 = elec2(charges, distances, idx_i, idx_j)

#plt.plot(distances, E1)
##plt.plot(distances, Eo1)
#plt.plot(distances, E2)
##plt.plot(distances, Eo2)
#plt.show()
#exit()

if False:

    #sampler = NormalModeScanner(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #sample_systems_optimize=True,
        #sample_systems_optimize_fmax=0.001,
        #)
    #sampler.run()

    #sampler = MDSampler(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #)
    #sampler.run()

    #sampler = MetaSampler(
        #sample_directory='test_samples',
        #sample_systems='data/hono.xyz',
        #sample_systems_format='xyz',
        #meta_cv=[[2, 1], [2, 3], [2, 0]],
        #meta_hookean=[[2, 1, 4.0], [2, 3, 4.0], [2, 0, 4.0]]
        #)
    #sampler.run()
    
    #sampler = MetaSampler(
        #sample_directory='test_samples',
        #sample_systems='data/co2.xyz',
        #sample_systems_format='xyz',
        #meta_cv=[[1, 0], [1, 2], [0, 1, 2]],
        #meta_gaussian_height=0.05,
        #meta_gaussian_widths=0.05,
        #meta_hookean=[[1, 0, 2.0], [1, 2, 2.0]],
        #meta_time_step=0.1,
        #meta_simulation_time=1.0E3,
        #meta_save_interval=100,
        #meta_initial_velocities=True,
        #meta_initial_temperature=300.
        #)
    #sampler.run()

    sampler = MetaSampler(
        sample_directory='test_samples',
        sample_systems='data/co2.xyz',
        sample_systems_format='xyz',
        sample_properties=['energy', 'forces', 'dipole'],
        meta_cv=[[1, 0], [1, 2], [0, 1, 2]],
        meta_gaussian_height=0.05,
        meta_gaussian_widths=0.05,
        meta_hookean=[[1, 0, 2.0], [1, 2, 2.0]],
        meta_time_step=0.1,
        meta_simulation_time=1.0E3,
        meta_save_interval=100,
        meta_initial_velocities=True,
        meta_initial_temperature=300.
        )
    sampler.run()


if False:
    
    recalculator = ReCalculator(
        recalc_interface='ase',
        recalc_calculator='ORCA',
        recalc_calculator_args={
            'orcasimpleinput': 'revPBE def2-TZVP',
            'charge': 0,
            'mult': 1,
            'directory': 'test_samples',
            #'orcablocks': '%pal nprocs 16 end'
            },
        recalc_properties=['energy', 'forces'],
        recalc_source_data_file='test_samples/1_meta.db',
        recalc_directory='test_samples'
        )
    recalculator.run()    
        
if True:

    data = DataContainer(
        data_file='data/fad_set3.db',
        data_source=[
            'data/fad.set3.58069.qmmm.mp2.avtz.npz'],
        data_load_properties=[
            'energy', 'force', 'total_charge', 'dipole'],
        data_unit_properties={
            'energy':   'eV',
            'forces':   'eV/Ang',
            'charge':   'e',
            'dipole':   'eAng'},
        data_alt_property_labels={
            'energy':   ['V', 'E']},
        data_overwrite=False)

if True:

    model = Asparagus(
        data_file='data/h2co_b3lyp.db',
        data_source=['data/h2co_B3LYP_cc_pVDZ_4001.npz'],
        data_load_properties=['energy', 'force', 'dipole'],
        model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        output_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
        trainer_optimizer_args={'lr': 0.0001},
        data_container=None)
    model.train()
    
    
