""" Converts the dataset samples from numpy to TFRecords format. """

import os

import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DEFAULT_DATASET_PATH = os.path.join("..", "data", "CREMA-D_Postprocessed")
DEFAULT_TFRECORDS_PATH = os.path.join("..", "data", "CREMA-D_TFRecords")
LOG_SPECS_KEY = "logspecs"
INPUT_TIME_DIM = 256 # Input dimensionality in the time axis
INPUT_FREQ_DIM = 128 # Input dimensionality in the frequency axis (number of bins)
INPUT_FEATURES_KEY = "input"
LABEL_KEY = "label"
ACTOR_KEY = "actor"
INPUT_DATA_DIR_NAME = "input_data"
LABEL_DIR_NAME = "labels"
ACTORS_DIR_NAME = "actors"


def get_arguments():
    """ Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Numpy to TFRecords.")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH,
                        help="Path to the directory with the dataset.")
    parser.add_argument("--tfrecords-path", type=str, default=DEFAULT_TFRECORDS_PATH,
                        help="Path where the TFRecords will be saved.")
    return parser.parse_args()


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    tfrecords_path = args.tfrecords_path

    if not os.path.exists(tfrecords_path):
        os.makedirs(tfrecords_path)

    input_dir_path = os.path.join(dataset_path, INPUT_DATA_DIR_NAME)
    labels_dir_path = os.path.join(dataset_path, LABEL_DIR_NAME)
    actors_dir_path = os.path.join(dataset_path, ACTORS_DIR_NAME)

    input_file_names = os.listdir(input_dir_path)
    label_file_names = os.listdir(labels_dir_path)
    actors_file_names = os.listdir(actors_dir_path)

    label_idx = 0
    for file_name in tqdm(input_file_names):
        if not file_name.endswith(".npz"):
            continue

        full_file_path = os.path.join(input_dir_path, file_name)
        X = np.load(full_file_path)[LOG_SPECS_KEY][:, :INPUT_TIME_DIM, :INPUT_FREQ_DIM]  # Loads a batch of inputs
        X = X.astype(np.float32)

        label_file_name = file_name.split('.')[0] + "_labels.npy"
        label_file_path = os.path.join(labels_dir_path, label_file_name)
        y = np.load(label_file_path)

        actor_file_name = file_name.split('.')[0] + "_actors.npy"
        actor_file_path = os.path.join(actors_dir_path, actor_file_name)
        actors = np.load(actor_file_path)

        for index, x in enumerate(X):
            label = y[index]
            actor = actors[index]
            tf_record_filename = file_name.split(".")[0] + "_" + str(index) + ".tfrecord"
            tf_record_fullpath = os.path.join(tfrecords_path, tf_record_filename)
            with tf.io.TFRecordWriter(tf_record_fullpath) as tfrecord_writer:
                feature = {
                    INPUT_FEATURES_KEY: _bytes_feature(tf.compat.as_bytes(x.tobytes())),
                    LABEL_KEY: _bytes_feature(bytes(label, encoding='ascii')),
                    ACTOR_KEY: _bytes_feature(bytes(actor, encoding='ascii'))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                tfrecord_writer.write(example.SerializeToString())
                tqdm.write("Saved record at: {}".format(tf_record_fullpath))