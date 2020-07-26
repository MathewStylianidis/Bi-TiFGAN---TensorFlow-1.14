import sys
import os

import numpy as np
import tensorflow as tf
import functools
from tqdm import tqdm

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import BiSpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools.data import fmap
from hyperparams.tifgan_hyperparams import get_hyperparams


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#path = os.path.join("/", "media", "datastore", "c-matsty-data", "SpeechCommands_Preproc_2_training", "input_data")
path = os.path.join("/", "media", "datastore", "c-matsty-data", "datasets",
                    "SpeechCommands", "SpeechCommands_Preproc_2_training", "input_data")

files = os.listdir(path)

print("Loading data")
X = []
for file in tqdm(files):
    if not file.endswith(".npz"):
        continue
    file_path = os.path.join(path, file)
    X.append(np.load(file_path)['logspecs'])
preprocessed_images = np.vstack(X)

#print(preprocessed_images.shape)
#print(np.max(preprocessed_images[:, :256, :]))
#print(np.min(preprocessed_images[:, :256, :]))
#print(np.mean(preprocessed_images[:, :256, :]))

dataset = Dataset(preprocessed_images[:, :256])


time_str = 'commands_md64_8k'
global_path = os.path.join("/", "media", "datastore", "c-matsty-data", "checkpoints_summaries",
                           "bitifgan-results-sc09-run1-nogp")

name = time_str

params = get_hyperparams(global_path, name)

resume, params = utils.test_resume(True, params)

biwgan = GANsystem(BiSpectrogramGAN, params)
print("Start training")
biwgan.train(dataset, resume=None)