"""
Evaluates the encoder of the Bi-GAN by calculating the reconstruction loss for a set of spectrograms for all different
checkpoints, feeding the spectrograms through the encoder and reconstructing them with the generator.
"""

__author__ = "Matthaios Stylianidis"

import os
import re

import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from feature_evaluation.utils import get_avg_spectrogram_reconstruction_error, load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '-1'

DEFAULT_NAME = "wgan"
DEFAULT_SAVE_DIR = os.path.join("..", "results")
DEFAULT_EPSILON = 0.0
INPUT_TIME_DIM = 256  # Input dimensionality in the time axis
INPUT_FREQ_DIM = 128  # Input dimensionality in the frequency axis (number of bins)
IMAGE_SIZE = INPUT_FREQ_DIM * INPUT_TIME_DIM


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate reconstruction error over time.")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to dataset with spectrograms to reconstruct.")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to the directory with all the checkpoints for the model.")
    parser.add_argument("--save-dir", type=str, required=False, default=DEFAULT_SAVE_DIR,
                        help="Path to the directory where results should be saved.")
    parser.add_argument("--model-name", type=str, required=False, default=DEFAULT_NAME,
                        help="The name of the trained model included in the checkpoint file names.")
    parser.add_argument("--epsilon", type=float, required=False, default=DEFAULT_EPSILON,
                        help="Epsilon to be used for a smoothness test.")
    return parser.parse_args()


def get_checkpoint_paths(checkpoints_dir, name):
    """ Gets a list of tuples with the update step values along with their corresponding checkpoint paths.

    Args:
        checkpoints_dir: The directory with all the checkpoints to be sequentially evaluated.
        name: The name of the trained model used in the checkpoint filenames.

    Returns:
        A list with tuples where the first element is an integer denoting the update step and the second is a string
            with the path to checkpoint for that update step
    """
    filenames = os.listdir(checkpoints_dir)
    # Remove extension from file names
    filenames = [filename.split('.')[0] for filename in filenames]
    # Remove duplicate names resulting from multiple files needed for the same checkpoint
    filenames = np.unique(filenames)

    # Get update step from each filename and create tuple list
    res = []
    update_step_pattern = "{}-[0-9]+".format(name)
    for filename in tqdm(filenames):
        try:
            step_string = re.findall(update_step_pattern, filename)[0]
            update_step = int(step_string.split('-')[-1])
            res.append((update_step, os.path.join(checkpoints_dir, filename)))
        except IndexError:
            # Exception will be caught for files that do not abide by the checkpoint format - files that we do not need
            continue

    # Sort tuple list according to update step by ascending order
    res.sort(key=lambda tup: tup[0])

    return res


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    checkpoint_dir = args.checkpoint_dir
    save_dir = args.save_dir
    name = args.model_name
    epsilon = args.epsilon

    print("-Getting and sorting checkpoint paths according to update step")
    checkpoint_tuples = get_checkpoint_paths(checkpoint_dir, name=name)

    print("-Loading dataset")
    X = load_data(dataset_path)

    print("-Start reconstructing spectrogram samples for each checkpoint")
    avg_reconstruction_error = get_avg_spectrogram_reconstruction_error(X, checkpoint_tuples, IMAGE_SIZE, epsilon)
    values = np.array(list(avg_reconstruction_error.values()))

    print("-Visualizing and saving result")
    title_size = 16
    label_size = 13
    title = "Spectrogram reconstruction error over time"
    log_title = "Spectrogram reconstruction error over time - Log scale"
    if epsilon != 0.0:
        title += " $\epsilon$ = {}".format(epsilon)
        log_title += " $\epsilon$ = {}".format(epsilon)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.plot(list(avg_reconstruction_error.keys()), values)
    plt.title(title, size=title_size)
    plt.xlabel("Update step", size=label_size)
    plt.ylabel("Reconstruction error", size=label_size)
    plt.savefig(os.path.join(save_dir, 'spectrogram_reconstruction_error_{}.png'.format(epsilon)))
    plt.cla()

    plt.plot(list(avg_reconstruction_error.keys()), np.log(values))
    plt.title(log_title, size=title_size)
    plt.xlabel("Update step", size=label_size)
    plt.ylabel("Log reconstruction error", size=label_size)
    plt.savefig(os.path.join(save_dir, 'log_spectrogram_reconstruction_error_{}.png'.format(epsilon)))
    plt.cla()













