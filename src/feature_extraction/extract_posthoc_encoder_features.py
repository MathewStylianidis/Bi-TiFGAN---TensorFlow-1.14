"""
Extracts features from a given dataset using a pre-trained Bi-GAN encoder as a feature extractor.
"""

__author__ = "Matthaios Stylianidis"

import os

import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from gantools.model import encoder
from hyperparams.tifgan_hyperparams import get_hyperparams, get_encoder_hyperparams

from feature_evaluation.utils import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to directory where the dataset is stored.")
    parser.add_argument("--checkpoint-path", type=str, required=False, default=None,
                        help="Path to the trained encoder checkpoint.")
    parser.add_argument("--features-path", type=str, required=True,
                        help="Path where the extracted features will be saved.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    checkpoint_path = args.checkpoint_path
    features_path = args.features_path

    X = load_data(dataset_path)

    params = get_hyperparams(result_path="", name="", bidirectional=False)
    batch_size = params["optimization"]["batch_size"]
    latent_dim = params["net"]["generator"]["latent_dim"]
    encoder_params = get_encoder_hyperparams(latent_dim, params["md"], params["bn"], params["input_shape"])

    features = []
    with tf.device('/gpu:0'):
        # Set up encoder
        X_placeholder = tf.placeholder(tf.float32, shape=[None, *params["input_shape"]])
        encoder_output = encoder(X_placeholder, encoder_params, reuse=False, scope="encoder")
        encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")

        # Create saver
        saver = tf.train.Saver(var_list=encoder_vars)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, checkpoint_path)

            for i in tqdm(range(0, len(X), batch_size)):
                x_batch = X[i:i+batch_size]
                z = sess.run(encoder_output, feed_dict={X_placeholder: x_batch})
                features.append(z)
        features = np.vstack(features)
        
        # Save features to designated path
        feat_dir_path = os.path.dirname(os.path.abspath(features_path))
        if not os.path.exists(feat_dir_path):
            os.makedirs(feat_dir_path)
        np.save(features_path, features)
        print("-Features saved at: {}".format(features_path))