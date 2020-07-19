"""
Extracts features from a given dataset using a pre-trained Bi-GAN encoder as a feature extractor.
"""

__author__ = "Matthaios Stylianidis"

import os

import tensorflow as tf
import numpy as np
import argparse

from gantools import data
from gantools import utils
from gantools import plot
from gantools import blocks
from gantools.model import BiSpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

time_str = 'commands_md64_8k'
global_path = 'saved_results'
name = time_str

INPUT_FEATURES_KEY = "input"
INPUT_TIME_DIM = 256  # Input dimensionality in the time axis
INPUT_FREQ_DIM = 128  # Input dimensionality in the frequency axis (number of bins)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--tfrecord-path", type=str, required=True,
                        help="Path to directory where the dataset (the tf_records) are stored.")
    parser.add_argument("--checkpoint-step", type=int, required=False, default=None,
                        help="Step of the checkpoint at which the Bi-TiF-GAN weights are saved.")
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path where the extracted features will be saved.")
    return parser.parse_args()


def parse_function(example_proto):
    features = {INPUT_FEATURES_KEY: tf.io.FixedLenFeature([], tf.string)}
    serialized_example = tf.io.parse_single_example(example_proto, features)
    input_features = serialized_example[INPUT_FEATURES_KEY]
    input_features = tf.io.decode_raw(input_features, tf.float32)
    input_features.set_shape([INPUT_TIME_DIM * INPUT_FREQ_DIM])
    input_features = tf.reshape(input_features, [INPUT_TIME_DIM, INPUT_FREQ_DIM, 1])
    return input_features


if __name__ == "__main__":
    args = get_arguments()
    tfrecord_path = args.tfrecord_path
    features_path = args.features_path
    checkpoint_step = args.checkpoint_step


    tf_dataset_filenames = os.listdir(tfrecord_path)
    tf_dataset_filepaths = [os.path.join(tfrecord_path, filename) for filename in tf_dataset_filenames]

    # Define GAN model
    bn = False
    md = 64
    latent_dim = 100
    input_shape = [256, 128, 1]
    downscale = 1
    batch_size = 64

    params_discriminator = dict()
    params_discriminator['stride'] = [2, 2, 2, 2, 2]
    params_discriminator['nfilter'] = [md, 2 * md, 4 * md, 8 * md, 16 * md]
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
    params_generator['nfilter'] = [8 * md, 4 * md, 2 * md, md, 1]
    params_generator['shape'] = [[12, 3], [12, 3], [12, 3], [12, 3], [12, 3]]
    params_generator['batch_norm'] = [bn, bn, bn, bn]
    params_generator['full'] = [256 * md]
    params_generator['summary'] = True
    params_generator['non_lin'] = tf.nn.tanh
    params_generator['activation'] = tf.nn.relu
    params_generator['data_size'] = 2
    params_generator['spectral_norm'] = True
    params_generator['in_conv_shape'] = [8, 4]

    params_optimization = dict()
    params_optimization['batch_size'] = batch_size
    params_optimization['epoch'] = 10000
    params_optimization['n_critic'] = 5
    params_optimization['generator'] = dict()
    params_optimization['generator']['optimizer'] = 'adam'
    params_optimization['generator']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
    params_optimization['generator']['learning_rate'] = 1e-4
    params_optimization['discriminator'] = dict()
    params_optimization['discriminator']['optimizer'] = 'adam'
    params_optimization['discriminator']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
    params_optimization['discriminator']['learning_rate'] = 1e-4
    params_optimization['encoder'] = dict()
    params_optimization['encoder']['optimizer'] = 'adam'
    params_optimization['encoder']['kwargs'] = {'beta1': 0.5, 'beta2': 0.9}
    params_optimization['encoder']['learning_rate'] = 1e-4

    # all parameters
    params = dict()
    params['net'] = dict()  # All the parameters for the model
    params['net']['generator'] = params_generator
    params['net']['discriminator'] = params_discriminator
    params['net']['encoder'] = params_encoder
    params['net']['prior_distribution'] = 'gaussian'
    params['net']['shape'] = input_shape  # Shape of the image
    params['net']['gamma_gp'] = 10  # Gradient penalty
    params['net']['fs'] = 16000 // downscale

    params['optimization'] = params_optimization
    params['summary_every'] = 100  # Tensorboard summaries every ** iterations
    params['print_every'] = 50  # Console summaries every ** iterations
    params['save_every'] = 1000  # Save the model every ** iterations
    params['save_dir'] = os.path.join(global_path, name + '_checkpoints/')
    params['Nstats'] = 500

    biwgan = GANsystem(BiSpectrogramGAN, params)

    X = []
    with tf.Session() as sess:
        dataset = tf.data.TFRecordDataset(tf_dataset_filepaths)
        dataset = dataset.map(parse_function)
        iterator = dataset.make_one_shot_iterator()
        input_sample = iterator.get_next()

        for path in tf_dataset_filepaths:
            x = sess.run(input_sample)
            X.append(x)
        print(len(X))
    X = np.stack(X)

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
