import sys
import os

import numpy as np
import tensorflow as tf
import ltfatpy
import argparse
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from gantools import data
from gantools import utils
from gantools import plot
from gantools.model import BiSpectrogramGAN
from gantools.data.Dataset import Dataset
from gantools.gansystem import GANsystem
from gantools import blocks
from gantools.data import fmap
from hyperparams.tifgan_hyperparams import get_hyperparams
from signal_processing.ltfat_stft import LTFATStft
from signal_processing.mob_gab_phase_grad import modgabphasegrad
from signal_processing.pghi import pghi


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
DEFAULT_OUTPUT_PATH = os.path.join("..", "generated_samples")
DEFAULT_NAME = "commands_md64_8k"
DEFAULT_SAMPLE_NO = 30

DEFAULT_FIXED_AUDIO_LEN = 16384
DEFAULT_FFT_HOP_SIZE = 128
DEFAULT_FFT_WINDOW_LEN = 512

ltfatpy.gabphasegrad = modgabphasegrad  # The original function is not implemented for one sided stfts on ltfatpy


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
                        help="Step of the checkpoint at which the Bi-TiF-GAN weights are saved.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing the training results for the model.")
    parser.add_argument("--output-path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path where the generated samples are saved.")
    parser.add_argument("--sample-no", type=int, default=DEFAULT_SAMPLE_NO,
                        help="Number of samples to generate.")
    parser.add_argument("--name", type=str, default=DEFAULT_NAME,
                        help="Name of the model.")
    parser.add_argument("--fixed-len", type=str, default=DEFAULT_FIXED_AUDIO_LEN,
                        help="Length of the generated audio samples.")
    parser.add_argument("--fft-hop-size", type=str, default=DEFAULT_FFT_HOP_SIZE,
                        help="Hop size when performing STFT on the signals.")
    parser.add_argument("--fft-window-length", type=str, default=DEFAULT_FFT_WINDOW_LEN,
                        help="Window size when performing STFT on the signals.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    results_dir = args.results_dir
    checkpoint_step = args.checkpoint_step
    gen_samples_dir_path = args.output_path
    nsamples = args.sample_no
    name = args.name
    fft_hop_size = args.fft_hop_size
    fft_window_length = args.fft_window_length
    L = args.fixed_len

    gen_samples_dir_path = os.path.join(gen_samples_dir_path, name)
    if not os.path.exists(gen_samples_dir_path):
        os.makedirs(gen_samples_dir_path)

    with tf.device('/gpu:0'):
        params = get_hyperparams(results_dir, name)
        biwgan = GANsystem(BiSpectrogramGAN, params)

        nlatent = params["net"]["generator"]["latent_dim"]

        # Generate latent variables
        d2 = clip_dist2(nsamples, nlatent)

        with tf.Session() as sess:
            biwgan.load(sess=sess, checkpoint=checkpoint_step)

            X_fake = sess.run(biwgan._net.X_fake, feed_dict={biwgan._net.z: d2})

            print("Generating samples")
            generated_spectrograms = np.squeeze(X_fake)
            generated_spectrograms = np.exp(5 * (generated_spectrograms - 1))  # Undo preprocessing
            generated_spectrograms = np.concatenate([generated_spectrograms, np.zeros_like(generated_spectrograms)[:, 0:1, :]],
                                               axis=1)  # Fill last column of freqs with zeros

            print("Phase recovery")
            anStftWrapper = LTFATStft()
            # Compute Tgrad and Fgrad from the generated spectrograms
            tgrads = np.zeros_like(generated_spectrograms)
            fgrads = np.zeros_like(generated_spectrograms)
            gs = {'name': 'gauss', 'M': fft_window_length}
            clipBelow = -10
            reconstructed_audios = np.zeros((len(generated_spectrograms), L))
            for index, magSpectrogram in enumerate(tqdm(generated_spectrograms)):
                tgrads[index], fgrads[index] = ltfatpy.gabphasegrad('abs', magSpectrogram, gs, fft_hop_size)
                logMagSpectrogram = np.log(magSpectrogram.astype(np.float64))
                phase = pghi(logMagSpectrogram, tgrads[index], fgrads[index], fft_hop_size, fft_window_length, L,
                             tol=10)
                reComplexStft = (np.e ** logMagSpectrogram) * np.exp(1.0j * phase)
                reComplexStft = reComplexStft[:257, :128]
                reconstructed_audio = anStftWrapper.inverseOneSidedStft(reComplexStft, fft_window_length, fft_hop_size)
                reconstructed_audios[index] = reconstructed_audio

            # Save spectrograms
            np.save(os.path.join(gen_samples_dir_path, name + "_" + str(checkpoint_step) + "_spectrograms.npy"),
                    generated_spectrograms)
            # Save audio signals as wav files
            for index, audio in tqdm(enumerate(reconstructed_audios)):
                path = os.path.join(gen_samples_dir_path, str(index) + "_" + name + "_" + str(checkpoint_step) + ".wav")
                wavfile.write(path, L, audio)
                tqdm.write("Stored generated audio file at: {}".format(path))
