import sys
sys.path.insert(0, '../../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import BiSpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
import functools
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy.io
import numpy as np
from tqdm import tqdm

downscale = 1

#path = os.path.join("/", "media", "datastore", "c-matsty-data", "SpeechCommands_Preproc_2_training", "input_data")
path = os.path.join("/", "media", "datastore", "c-matsty-data", "SpeechCommands_Preproc_21training", "input_data")

files = os.listdir(path)

print("Loading data")
X = []
for file in tqdm(files):
    if not file.endswith(".npz"):
        continue
    file_path = os.path.join(path, file)
    X.append(np.load(file_path)['logspecs'])
preprocessed_images = np.vstack(X)
print(preprocessed_images.shape)

print(np.max(preprocessed_images[:, :256, :]))
print(np.min(preprocessed_images[:, :256, :]))
print(np.mean(preprocessed_images[:, :256, :]))

dataset = Dataset(preprocessed_images[:, :256])


time_str = 'commands_md64_8k'
global_path = '../../bitifgan-_results-sc09-run2'

name = time_str

from gantools import blocks
bn = False

md = 64
latent_dim = 100
input_shape = [256, 128, 1]

params_discriminator = dict()
params_discriminator['stride'] = [2,2,2,2,2]
params_discriminator['nfilter'] = [md, 2*md, 4*md, 8*md, 16*md]
params_discriminator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
params_discriminator['batch_norm'] = [bn, bn, bn, bn, bn]
params_discriminator['latent_full'] = [50]
params_discriminator['latent_activation'] = blocks.lrelu
params_discriminator['full'] = []
params_discriminator['minibatch_reg'] = False
params_discriminator['summary'] = True
params_discriminator['data_size'] = 2
params_discriminator['apply_phaseshuffle'] = True
params_discriminator['spectral_norm'] = True
params_discriminator['activation'] = blocks.lrelu

params_encoder = dict()
params_encoder['input_shape'] = input_shape  # Shape of the image
params_encoder['stride'] = [2, 2, 2, 2, 2]
params_encoder['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
params_encoder['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
params_encoder['batch_norm'] = [bn, bn, bn, bn, bn]
params_encoder['full'] = []
params_encoder['summary'] = True
params_encoder['data_size'] = 2
params_encoder['apply_phaseshuffle'] = True
params_encoder['spectral_norm'] = True
params_encoder['activation'] = blocks.lrelu
params_encoder['latent_dim'] = latent_dim  # Dimensionality of the latent representation

params_generator = dict()
params_generator['stride'] = [2, 2, 2, 2, 2]
params_generator['latent_dim'] = latent_dim
params_generator['consistency_contribution'] = 0
params_generator['nfilter'] = [8*md, 4*md, 2*md, md, 1]
params_generator['shape'] = [[12, 3],[12, 3], [12, 3],[12, 3],[12, 3]]
params_generator['batch_norm'] = [bn, bn, bn, bn]
params_generator['full'] = [256*md]
params_generator['summary'] = True
params_generator['non_lin'] = tf.nn.tanh
params_generator['activation'] = tf.nn.relu
params_generator['data_size'] = 2
params_generator['spectral_norm'] = True 
params_generator['in_conv_shape'] = [8, 4]

params_optimization = dict()
params_optimization['batch_size'] = 64
params_optimization['epoch'] = 10000
params_optimization['n_critic'] = 5
params_optimization['generator'] = dict()
params_optimization['generator']['optimizer'] = 'adam'
params_optimization['generator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['generator']['learning_rate'] = 1e-4
params_optimization['discriminator'] = dict()
params_optimization['discriminator']['optimizer'] = 'adam'
params_optimization['discriminator']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['discriminator']['learning_rate'] = 1e-4
params_optimization['encoder'] = dict()
params_optimization['encoder']['optimizer'] = 'adam'
params_optimization['encoder']['kwargs'] = {'beta1':0.5, 'beta2':0.9}
params_optimization['encoder']['learning_rate'] = 1e-4


# all parameters
params = dict()
params['net'] = dict() # All the parameters for the model
params['net']['generator'] = params_generator
params['net']['discriminator'] = params_discriminator
params['net']['encoder'] = params_encoder
params['net']['prior_distribution'] = 'gaussian'
params['net']['shape'] = input_shape # Shape of the image
params['net']['gamma_gp'] = 10  # Gradient penalty
params['net']['fs'] = 16000//downscale

params['optimization'] = params_optimization
params['summary_every'] = 100  # Tensorboard summaries every ** iterations
params['print_every'] = 50  # Console summaries every ** iterations
params['save_every'] = 1000  # Save the model every ** iterations
params['summary_dir'] = os.path.join(global_path, name +'_summary/')
params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
params['Nstats'] = 500

resume, params = utils.test_resume(True, params)
params['optimization']['epoch'] = 10000

biwgan = GANsystem(BiSpectrogramGAN, params)
print("Start training")
biwgan.train(dataset, resume=None)