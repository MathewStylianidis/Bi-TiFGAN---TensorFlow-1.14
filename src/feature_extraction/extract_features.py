"""
Extracts features from a given dataset using a pre-trained Bi-GAN encoder as a feature extractor.
"""

__author__ = "Matthaios Stylianidis"

import os

import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from gantools import data
from gantools import utils
from gantools import plot
from gantools import blocks
from gantools.model import BiSpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from hyperparams.tifgan_hyperparams import get_hyperparams

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time_str = 'commands_md64_8k'
global_path = 'saved_results'
name = time_str


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to directory where the dataset (the tf_records) are stored.")
    parser.add_argument("--checkpoint-step", type=int, required=False, default=None,
                        help="Step of the checkpoint at which the Bi-TiF-GAN weights are saved.")
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path where the extracted features will be saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    checkpoint_step = args.checkpoint_step
    features_path = args.features_path

    print("Loading data")
    X = []
    for file in tqdm(files):
        if not file.endswith(".npz"):
            continue
        file_path = os.path.join(path, file)
        X.append(np.load(file_path)['logspecs'])
    X = np.vstack(X)

    params = get_hyperparams(global_path, name)
    biwgan = GANsystem(BiSpectrogramGAN, params)

    features = []
    with tf.Session() as sess:
        biwgan.load(sess=sess, checkpoint=checkpoint_step)

        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            z = sess.run(biwgan._net.z_real, feed_dict={biwgan._net.X_real: x_batch})
            features.append(z)
    features = np.vstack(features)

    # Save features to designated path
    feat_dir_path = os.path.dirname(os.path.abspath(features_path))
    if not os.path.exists(feat_dir_path):
        os.makedirs(feat_dir_path)
    np.save(features_path, features)
    print("-Features saved at: {}".format(features_path))
