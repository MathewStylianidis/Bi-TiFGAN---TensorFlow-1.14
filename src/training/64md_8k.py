import sys
import os

import numpy as np
import tensorflow as tf
import functools
import argparse
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

DEFAULT_DATASET_PATH = os.path.join("/", "media", "datastore", "c-matsty-data", "datasets",
                    "SpeechCommands", "SpeechCommands_Preproc_2_training", "input_data")
DEFAULT_RESULTS_PATH = os.path.join("/", "media", "datastore", "c-matsty-data", "checkpoints_summaries")
DEFAULT_NAME = 'commands_md64_8k'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a bi-directional TiF-GAN.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path where the training dataset is saved.")
    parser.add_argument("--results-path", type=str, default=DEFAULT_RESULTS_PATH,
                        help="Path where the checkpoints and the event summary are saved.")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="Name of the model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    path = args.dataset_path
    global_path = args.results_path
    name = args.name

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

    params = get_hyperparams(global_path, name)

    resume, params = utils.test_resume(True, params)

    biwgan = GANsystem(BiSpectrogramGAN, params)
    print("Start training")
    biwgan.train(dataset, resume=None)