import sys
import os

import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm

from gantools.model import SpectrogramGAN, encoder
from gantools.gansystem import GANsystem
from hyperparams.tifgan_hyperparams import get_hyperparams, get_encoder_hyperparams


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEFAULT_ENCODER_PATH = os.path.join("..", "encoder_training_results")
DEFAULT_NAME = "commands_md64_8k"
DEFAULT_UPDATE_STEPS = 100000


def clip_dist2(nsamples, nlatent, m=2.5):
    shape = [nsamples, nlatent]
    z = np.random.randn(*shape)
    support = np.logical_or(z < -m, z > m)
    while np.sum(support):
        z[support] = np.random.randn(*shape)[support]
        support = np.logical_or(z < -m, z > m)
    return z


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a bi-directional TiF-GAN.")
    parser.add_argument("--checkpoint-step", type=int, required=False, default=None,
                        help="Step of the checkpoint at which the TiF-GAN weights are saved.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing the training results for the model.")
    parser.add_argument("--encoder-path", type=str, default=DEFAULT_ENCODER_PATH,
                        help="Path where the encoder's training results will be saved.")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="Name of the model.")
    parser.add_argument("--update-steps", type=int, default=DEFAULT_UPDATE_STEPS,
                        help="Number of update steps for the encoder")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    results_dir = args.results_dir
    checkpoint_step = args.checkpoint_step
    encoder_path = args.encoder_path
    update_steps = args.update_steps
    name = args.name

    with tf.device('/gpu:0'):
        # Build GAN
        params = get_hyperparams(results_dir, name, bidirectional=False)
        gan = GANsystem(SpectrogramGAN, params)

        batch_size = params["optimization"]["batch_size"]
        latent_dim = params["net"]["generator"]["latent_dim"]

        # Build encoder
        # Define placeholder for variable z
        X_placeholder = tf.placeholder(tf.float32, shape=[None, *params["input_shape"]])
        z_placeholder = tf.placeholder(tf.float32, shape=[None, 100])
        encoder_params = get_encoder_hyperparams(latent_dim, params["md"], params["bn"], params["input_shape"])
        encoder_output = encoder(X_placeholder, encoder_params, reuse=False, scope="encoder")
        # Set up loss
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        mse = tf.losses.mean_squared_error(z_placeholder, encoder_output)
        encoder_opt = tf.train.AdamOptimizer(1e-4).minimize(mse, var_list=encoder_vars)

        with tf.Session() as sess:
            gan.load(sess=sess, checkpoint=checkpoint_step)

            print("Starting encoder training...")
            for i in tqdm(range(update_steps)):
                # Generate latent variables
                z = clip_dist2(batch_size, latent_dim)
                # Generate spectrograms
                X_fake = sess.run(gan._net.X_fake, feed_dict={gan._net.z: z})
                # Train encoder to reconstruct spectrograms
                loss = sess.run(encoder_opt, feed_dict={X_placeholder: X_fake, z_placeholder: z})
