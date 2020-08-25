"""
Utility function for the different evaluation scripts.
"""

__author__ = "Matthaios Stylianidis"

import os
import sys

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from gantools.model import BiSpectrogramGAN
from gantools.gansystem import GANsystem
from hyperparams.tifgan_hyperparams import get_hyperparams


def get_avg_reconstruction_error(Z, checkpoint_tuples, epsilon=0.0, batch_size=64):
    """ Calculates the mean absolute reconstruction error for every checkpoint.

    Args:
        Z (np.array): The dataset_size x latent_dimensions numpy array matrix with the latent samples to be
            fed through the generator and the encoder to calculate the reconstruction error.
        checkpoint_tuples (list): A list of tuples where each tuple contains the update step and the path to
            the directory with the checkpoints.
        epsilon (float): A number to be added to the latent samples before feeding them to the generator. This
            argument can be set to small non zero values to evaluate the smoothness of the function learned by
            the generator and the encoder. Defaults to 0.0.
        batch_size (int): The number of latent to feed each time in parallel to the generator and encoder.

    Returns:
        A dictionary where the key is the update step corresponding to a checkpoint and the value is the
            mean absolute reconstruction error acquired with that checkpoint.
    """
    avg_reconstruction_error = {}
    dataset_size = Z.shape[0]

    for update_step, checkpoint_path in tqdm(checkpoint_tuples):
        # Get the parent directory to the checkpoint directory
        results_path = os.path.join(checkpoint_path, "..", "..")

        with tf.device('/gpu:0'):
            params = get_hyperparams(results_path, "commands_md64_8k")
            # Block print
            sys.stdout = open(os.devnull, 'w')
            biwgan = GANsystem(BiSpectrogramGAN, params)

            with tf.Session() as sess:
                biwgan.load(sess=sess, checkpoint=update_step)
                # Enable print
                sys.stdout = sys.__stdout__

                cum_rec_error = 0
                for i in range(0, len(Z), batch_size):
                    z_batch = Z[i:i + batch_size]
                    x_batch = sess.run(biwgan._net.X_fake, feed_dict={biwgan._net.z: z_batch + epsilon})
                    z_batch_hat = sess.run(biwgan._net.z_real, feed_dict={biwgan._net.X_real: x_batch})
                    # Compute sum of absolute errors for all latent samples
                    cum_rec_error += np.sum(np.abs(z_batch - z_batch_hat), axis=1).sum()

                # Divide the sum of all absolute errors by the dataset size to get the mean error
                avg_reconstruction_error[update_step] = cum_rec_error / dataset_size

    return avg_reconstruction_error


def load_data_labels(labels_path):
    """ Loads the label identifiers for all dataset samples.

    Args:
        labels_path: The path to the directory with the stored labels.

    Returns:
        A numpy array with the dataset labels.

    """
    files = os.listdir(labels_path)
    Y = []
    for file in tqdm(files):
        if not file.endswith(".npy"):
            continue
        file_path = os.path.join(labels_path, file)
        Y.append(np.load(file_path).reshape((-1, 1)))
    Y = np.vstack(Y).flatten()
    return Y




