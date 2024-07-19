
import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')

# Start training a default PhysNet model (model_type='physnet' [default]).
from asparagus import Asparagus
model = Asparagus(
    config='clphoh_physnet.json',
    data_file='clphoh.db',
    data_source=[
        'data/meta-CL-phenol_6-31G_MP2_25000.npz',
        'data/ortho-CL-phenol_6-31G_MP2_25000.npz',
        'data/para-CL-phenol_6-31G_MP2_25000.npz'],
    model_type='physnet',
    model_directory='model_clphoh',
    model_properties=['energy', 'forces', 'dipole'],
    trainer_max_epochs=1_000,
    trainer_guess_shifts=True,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
