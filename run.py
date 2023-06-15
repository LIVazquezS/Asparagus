
from asparagus import DataContainer

# from asparagus import Sampler, NormalModeScanner, MDSampler

from asparagus import Asparagus

# if False:
#
#     data = DataContainer(
#         data_file='data/fad_set3.db',
#         data_source=[
#             'data/fad.set3.58069.qmmm.mp2.avtz.npz'],
#         data_load_properties=[
#             'energy', 'force', 'total_charge', 'dipole'],
#         data_unit_properties={
#             'energy':   'eV',
#             'forces':   'eV/Ang',
#             'charge':   'e',
#             'dipole':   'eAng'},
#         data_alt_property_labels={
#             'energy':   ['V', 'E']},
#         data_overwrite=False)
#
# if True:
#
#     #sampler = NormalModeScanner(
#         #sample_directory='test_samples',
#         #sample_systems='data/hono.xyz',
#         #sample_systems_format='xyz',
#         #sample_systems_optimize=True,
#         #sample_systems_optimize_fmax=0.001,
#         #)
#     #sampler.run()
#
#     sampler = MDSampler(
#         sample_directory='test_samples',
#         sample_systems='data/hono.xyz',
#         sample_systems_format='xyz',
#         )
#     sampler.run()
#
# if False:
#
model = Asparagus(data_file='data/h2co_b3lyp.db',
                  data_source=['data/h2co_B3LYP_cc_pVDZ_4001.npz'],
                  data_load_properties=['energy', 'force', 'dipole'],
                  model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
                  output_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
                  trainer_optimizer_args={'lr': 0.0001},
                  data_container=None)
#
ckpt = '20230525163715_YQO817Vk_F128KNoneb5a2i3o1cut8.0eTruedTruerFalse/best/best_model.pt'
model.test_model(ckpt,plot=True,show_plots=True,residual_plots=True,show_residuals=True,histogram_plots=True,show_histograms=True)
