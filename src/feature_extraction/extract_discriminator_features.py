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
from gantools.model import SpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from hyperparams.tifgan_hyperparams import get_hyperparams
from feature_evaluation.utils import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

name = 'commands_md64_8k'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to directory where the input data is stored.")
    parser.add_argument("--checkpoint-step", type=int, required=False, default=None,
                        help="Step of the checkpoint at which the Bi-TiF-GAN weights are saved.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing the training results for the model.")
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path where the extracted features will be saved.")
    parser.add_argument("--selected-layer", type=int, required=False, default=-1,
                        help="The index of the convolutional layer of the discriminator to use for extracting" +\
                             "the features")
    return parser.parse_args()


def extract_discriminator_features(X, results_dir, checkpoint_step, name, selected_layer, batch_size=64):
    """ Loads a discriminator and extracts features for a given dataset

    Args:
        X: Numpy array with the input images.
        results_dir: Directory containing the training results for the model.
        checkpoint_step: Step of the checkpoint at which the Bi-TiF-GAN weights are saved.
        name: The name of the trained model used in the checkpoint filenames.
        selected_layer: The index of the convolutional layer of the discriminator to use for extracting the features.
        batch_size: The number of images that will be processed at a time by the discriminator.

    Returns:
        The extracted features for X as a numpy array.
    """
    with tf.device('/gpu:0'):
        params = get_hyperparams(results_dir, name, bidirectional=False)
        tifgan = GANsystem(SpectrogramGAN, params)

        features = []
        with tf.Session() as sess:
            tifgan.load(sess=sess, checkpoint=checkpoint_step)

            for i in tqdm(range(0, len(X), batch_size)):
                x_batch = X[i:i + batch_size]
                feats = sess.run(tifgan._net.discr_features_real[selected_layer],
                                 feed_dict={tifgan._net.X_real: x_batch})
                features.append(feats)
        features = np.vstack(features)

    return features


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    results_dir = args.results_dir
    checkpoint_step = args.checkpoint_step
    features_path = args.features_path
    selected_layer = args.selected_layer

    X = load_data(dataset_path)

    features = extract_discriminator_features(X, results_dir, checkpoint_step, name, selected_layer)
    print(features.shape)
    # Save features to designated path
    feat_dir_path = os.path.dirname(os.path.abspath(features_path))
    if not os.path.exists(feat_dir_path):
        os.makedirs(feat_dir_path)
    np.save(features_path, features)
    print("-Features saved at: {}".format(features_path))