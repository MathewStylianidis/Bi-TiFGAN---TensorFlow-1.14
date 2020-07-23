"""
 Preprocesses a given dataset and saves the result in the given directory.
"""

__author__ = "Matthaios Stylianidis"

import os

import argparse

import preprocessing.utils as preprocessing_utils

DEFAULT_PREPROC_FILENAME = "postprocessed_data"
DEFAULT_PREPROC_TYPE = 1  # High-energy cropping is the default
DEFAULT_FIXED_AUDIO_LEN = 16384  # Length to which all signals must be resized to with the preprocessing
DEFAULT_CHUNK_OVERLAP = 1024  # Sample chunk overlap
DEFAULT_FFT_HOP_SIZE = 128
DEFAULT_FFT_WINDOW_LEN = 512
DEFAULT_BATCH_SIZE = 128
INPUT_DATA_DIR_NAME = "input_data"
LABEL_DIR_NAME = "labels"
ACTORS_DIR_NAME = "actors"
TIFGAN_FEATURES = "tifgan"
MFCC_FEATURES = "mfcc"
FBANK_FEATURES = "fbank"
dataset_names = preprocessing_utils.dataset_names


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to directory where the dataset is stored.")
    parser.add_argument("--results-path", type=str, required=True,
                        help="Path where the preprocessed dataset files will be saved.")
    parser.add_argument("--preproc-filename", type=str, default=DEFAULT_PREPROC_FILENAME,
                        help="Prefix of the name of the files with the preprocessed data.")
    parser.add_argument("--preproc-type", type=str, default=DEFAULT_PREPROC_TYPE,
                        help="Code for the type of the preprocessing to be used for fixing the input size.")
    parser.add_argument("--fixed-len", type=str, default=DEFAULT_FIXED_AUDIO_LEN,
                        help="Size of the preprocessed samples.")
    parser.add_argument("--chunk-overlap", type=str, default=DEFAULT_CHUNK_OVERLAP,
                        help="Overlap between chunks when samples are broken down to chunks.")
    parser.add_argument("--fft-hop-size", type=str, default=DEFAULT_FFT_HOP_SIZE,
                        help="Hop size when performing STFT on the signals.")
    parser.add_argument("--fft-window-length", type=str, default=DEFAULT_FFT_WINDOW_LEN,
                        help="Window size when performing STFT on the signals.")
    parser.add_argument("--batch-size", type=str, default=DEFAULT_BATCH_SIZE,
                        help="Number of samples in each batch saved in the preprocessed data directory.")
    parser.add_argument("--features-type", type=str, default=TIFGAN_FEATURES,
                        choices=[TIFGAN_FEATURES, MFCC_FEATURES, FBANK_FEATURES],
                        help="The type of features to be extracted from the recorded audio.")
    parser.add_argument("--dataset-name", type=str, required=False, choices=dataset_names, default=dataset_names[0],
                        help="The name of the dataset to read.")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess_types = [
        preprocessing_utils.CHUNK_CODE,
        preprocessing_utils.HIGH_ENERGY_CODE,
        preprocessing_utils.CROP_OR_PAD_CODE
    ]

    args = get_arguments()
    dataset_path = args.dataset_path
    results_path = args.results_path
    preproc_filename = args.preproc_filename
    preproc_type = preprocess_types[int(args.preproc_type)]
    fixed_len = args.fixed_len
    chunk_overlap = args.chunk_overlap
    fft_hop_size = args.fft_hop_size
    fft_window_length = args.fft_window_length
    batch_size = args.batch_size
    features_type = args.features_type
    dataset_name = args.dataset_name

    input_dir_path = os.path.join(results_path, INPUT_DATA_DIR_NAME)
    labels_dir_path = os.path.join(results_path, LABEL_DIR_NAME)
    actors_dir_path = os.path.join(results_path, ACTORS_DIR_NAME)
    if not os.path.exists(input_dir_path):
        os.makedirs(input_dir_path)
    if not os.path.exists(labels_dir_path):
        os.makedirs(labels_dir_path)
    if not os.path.exists(actors_dir_path):
        os.makedirs(actors_dir_path)


    X, y, actors, sr = preprocessing_utils.read_dataset(dataset_name, dataset_path, preproc_type, fixed_len,
                                                        chunk_overlap)
    print("Number of samples in the dataset: " + str(X.shape[0]))

    if features_type == TIFGAN_FEATURES:
        print("Applying the transformations from the time to the frequency domain...")
        preprocessing_utils.extract_tifgan_features(X, y, actors, fft_hop_size, fft_window_length, batch_size, input_dir_path,
                            labels_dir_path, actors_dir_path, preproc_filename)
    elif features_type == MFCC_FEATURES:
        preprocessing_utils.exctract_mfcc_features(X, y, actors, fft_window_length, fft_hop_size, sr, batch_size,
                                                   input_dir_path, labels_dir_path, actors_dir_path, preproc_filename)
    elif features_type == FBANK_FEATURES:
        preprocessing_utils.exctract_mfcc_features(X, y, actors, fft_window_length, fft_hop_size, sr, batch_size,
                                                   input_dir_path, labels_dir_path, actors_dir_path, preproc_filename,
                                                   fbank_feats=True)
