"""
 Script for splitting a dataset into a training and a test set.

 This script takes the files from a dataset directory and splits it into two separate directories, one with the
 training set and one with the test set, optionally replacing completely the original directory.
"""

__author__ = "Matthaios Stylianidis"

import os
import shutil

import argparse

from preprocessing.utils import filelines_to_list
from tqdm import tqdm


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to directory where the dataset is stored.")
    parser.add_argument('--test-list-path', type=str, required=False,
                        help="Path to the file with the paths to the test files.")
    parser.add_argument('--replace-dir', action='store_true')
    return parser.parse_args()


def ig_f(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]


def split_dataset(dataset_path, test_set_paths, replace_dir):
    """ Splits dataset into a training set and test set.

    Args:
        dataset_path (str): The path to the dataset.
        test_set_paths (list): The list with the test file paths.
        replace_dir (bool): Whether to delete the original dataset directory or not.

    Returns:

    """
    dataset_root_dir = os.path.dirname(dataset_path)
    dataset_dir_name = os.path.basename(dataset_path)
    training_set_path = os.path.join(dataset_root_dir, dataset_dir_name + "_training")
    test_set_path = os.path.join(dataset_root_dir, dataset_dir_name + "_test")

    ## Create directories for training and test set
    shutil.copytree(dataset_path, training_set_path, ignore=ig_f)
    shutil.copytree(dataset_path, test_set_path, ignore=ig_f)

    dataset_paths = []
    # Get list of dataset file paths
    for root, subdirs, files in os.walk(dataset_path):
        dir_name = os.path.basename(root)
        file_paths = [os.path.join(dir_name, file_name) for file_name in os.listdir(root)
                      if not os.path.isdir(os.path.join(root, file_name))]
        dataset_paths.extend(file_paths)

    test_set_paths = [path for path in test_set_paths if path in dataset_paths]
    training_set_paths = [path for path in dataset_paths if path not in test_set_paths]

    print("Creating test dataset...")
    for file_path in tqdm(test_set_paths):
        src = os.path.join(dataset_path, file_path)
        dest = os.path.join(test_set_path, file_path)
        shutil.copy(src, dest)

    print("Creating training dataset...")
    for file_path in tqdm(training_set_paths):
        src = os.path.join(dataset_path, file_path)
        dest = os.path.join(training_set_path, file_path)
        shutil.copy(src, dest)

    if replace_dir:
        shutil.rmtree(dataset_path)


if __name__ == "__main__":
    args = get_arguments()
    dataset_path = args.dataset_path
    test_list_path = args.test_list_path
    replace_dir = args.replace_dir

    # Read paths to test files
    test_set_paths = filelines_to_list(test_list_path)
    test_set_paths = [file_path.strip() for file_path in test_set_paths]

    split_dataset(dataset_path, test_set_paths, replace_dir)

