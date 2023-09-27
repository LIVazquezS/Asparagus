import numpy as np
import torch
import tensorflow as tf
from NNCalculator.neural_network.NeuralNetwork import *
# import argparse
#
# parser = argparse.ArgumentParser(description='Convert tensorflow model to pytorch model')
# parser.add_argument('--checkpoint', type=str, help='path to tensorflow checkpoint')
# parser.add_argument('--output', type=str, help='path to output pytorch model')
# args = parser.parse_args()
#
# print('The initial tensorflow model: {}'.format(args.checkpoint))
# print('The output pytorch model: {}'.format(args.output))

# This part is needed to load the tensorflow model

tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

checkpoint = "hf_models/betadiketones_hf_vdz_71208_a"
nn = NeuralNetwork(F=128,K=64,num_blocks=5,num_residual_atomic=2,num_residual_interaction=3,num_residual_output=1,
                   sr_cut=10.0,scope="neural_network")

session = tf.compat.v1.Session()

nn.restore(session, checkpoint)

trainable_vars = tf.compat.v1.trainable_variables()

variable_values = session.run(trainable_vars)

dct_tensorflow = {}
for var, value in zip(trainable_vars, variable_values):
    dct_tensorflow[var.name] = value

# This part loads the equivalences between the tensorflow and pytorch variables

dct_equivalences = np.load('parameters_convertion.npy', allow_pickle=True)

# print(dct_equivalences.item())

dct_to_torch = {}
for i,j in enumerate(dct_equivalences.item()):
    dct_to_torch[dct_equivalences.item().get(j)] = dct_tensorflow[j]

del dct_to_torch['atomic_charges_scaling']
del dct_to_torch['atomic_energies_scaling']

# To create the energy and charge scaling parameters

Eshift = dct_tensorflow['neural_network/Eshift:0']
Escale = dct_tensorflow['neural_network/Escale:0']

Qscale = dct_tensorflow['neural_network/Qscale:0']
Qshift = dct_tensorflow['neural_network/Qshift:0']

energy_scaling = []
for i,j in enumerate(Eshift):
    energy_scaling.append([Escale[i], j])

charge_scaling = []
for i,j in enumerate(Qshift):
    charge_scaling.append([Qscale[i], j])

dct_to_torch['atomic_energies_scaling'] = np.array(energy_scaling)
dct_to_torch['atomic_charges_scaling'] = np.array(charge_scaling)

# There are some parameters that were not used in the tensorflow model, so we define them equal to the standard values
# or equal to zero

dct_to_torch['electrostatic_model.switch_fn.cutoff'] = 14.0
dct_to_torch['input_model.input_cutoff_fn.cutoff'] = 14.0
dct_to_torch['input_model.input_descriptor_fn.rbf_cutoff_fn.cutoff'] = 14.0

dct_to_torch['input_model.atom_features'] = np.zeros((95, 128))

for i in dct_to_torch.keys():
    dct_to_torch[i] = torch.tensor(dct_to_torch[i])

torch.save({'model_state_dict': dct_to_torch}, 'best_model.pt')



