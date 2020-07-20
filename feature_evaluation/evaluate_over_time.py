"""
Evaluates the features learned by the Bi-GAN over its training time by loading each checkpoint and training a model
on them for a supervised task. The script shows a plot of the performance of the model over time but can also save
the model's performance in a file.
"""

__author__ = "Matthaios Stylianidis"

import os
import re
import random
import subprocess

import tensorflow as tf
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_NAME = "wgan"
INPUT_FEATURES_KEY = "input"
LABEL_KEY = "label"
ACTOR_KEY = "actor"
INPUT_TIME_DIM = 256  # Input dimensionality in the time axis
INPUT_FREQ_DIM = 128  # Input dimensionality in the frequency axis (number of bins)
DEFAULT_SAVE_DIR = os.path.join("..", "results")

# Define strings that denote the different sk-learn models to be used for feature evaluation
LOGISTIC_REGRESSION_STR = "LogisticRegression"
RANDOM_FOREST_STR = "RandomForest"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--tfrecord-path-train", type=str, required=True,
                        help="Path to directory where the training dataset (the tf_records) is stored.")
    parser.add_argument("--tfrecord-path-test", type=str, required=True,
                        help="Path to directory where the test dataset (the tf_records) is stored.")
    parser.add_argument("--checkpoints-dir", type=str, required=True,
                        help="Path to the directory with all the checkpoints for the model.")
    parser.add_argument("--save-dir", type=str, required=False, default=DEFAULT_SAVE_DIR,
                        help="Path to the directory where results should be saved.")
    parser.add_argument("--evaluation-model", type=str, required=False, default=LOGISTIC_REGRESSION_STR,
                        help="String denoting the type of sklearn model to be used for evaluating the features.")
    parser.add_argument("--model-name", type=str, required=False, default=DEFAULT_NAME,
                        help="The name of the trained model included in the checkpoint file names.")
    return parser.parse_args()


def parse_function(example_proto):
    features = {
        INPUT_FEATURES_KEY: tf.io.FixedLenFeature([], tf.string),
        LABEL_KEY: tf.io.FixedLenFeature([], tf.string),
        ACTOR_KEY: tf.io.FixedLenFeature([], tf.string)
    }
    serialized_example = tf.io.parse_single_example(example_proto, features)
    input_features = serialized_example[INPUT_FEATURES_KEY]
    input_features = tf.io.decode_raw(input_features, tf.float32)
    input_features.set_shape([INPUT_TIME_DIM * INPUT_FREQ_DIM])
    input_features = tf.reshape(input_features, [INPUT_TIME_DIM, INPUT_FREQ_DIM, 1])

    label = serialized_example[LABEL_KEY]
    label = tf.io.decode_raw(label, tf.uint8)

    actor = serialized_example[ACTOR_KEY]
    actor = tf.io.decode_raw(actor, tf.uint8)

    return input_features, label, actor


def load_data_labels(tf_record_path):
    """ Loads the labels and the actor identifiers for each sample in the dataset.

    Args:
        tf_record_path: The path to the directory with the TFRecrods.

    Returns:
        A list of tuples of (label, actor-id) for each sample in the dataset.

    """
    with tf.Session() as sess:
        tf_dataset_filenames = os.listdir(tf_record_path)
        tf_dataset_filepaths = [os.path.join(tf_record_path, filename) for filename in tf_dataset_filenames]
        tfrecord_dataset = tf.data.TFRecordDataset(tf_dataset_filepaths)
        tfrecord_dataset = tfrecord_dataset.map(parse_function, num_parallel_calls=1)
        iterator = tfrecord_dataset.make_one_shot_iterator()
        x = iterator.get_next()

        y = []
        actors = []
        for _ in range(len(tf_dataset_filepaths)):
            features, label, actor = sess.run(x)
            label = str(label.flatten(), 'ascii')
            y.append(label)
            actor = str(actor.flatten(), 'ascii')
            actors.append(actor)

    y = np.stack(y)
    actors = np.stack(actors)
    return y, actors


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


def visualize(results, label_dict, save_dir, avg_dataset_norms):
    """ Visualizes the results of the feature extraction and evaluation over the different update steps.

    Args:
        results: A list of tuples where each tuple is the update step number along with a numpy array containing
            the precision, recall and f1-score for each class.
        label_dict: A dictionary mapping the index of each class to its string representation.
        save_dir: The directory where the results should be saved.

    Returns:
        None
    """
    n_classes = len(label_dict.keys())

    class_metric = {i: {} for i in range(n_classes)}
    for i in range(n_classes):
        class_metric[i]["precision"] = []
        class_metric[i]["recall"] = []
        class_metric[i]["f1_score"] = []

    avg_recall_list = []
    avg_precision_list = []
    avg_f1_score_list = []
    update_steps = []
    for update_step, metrics in results:
        acc_recall = 0
        acc_precision = 0
        acc_f1_score = 0
        # For each class
        for i in range(len(metrics)):
            # Get precision, recall and f-1 score
            class_metric[i]["precision"].append(metrics[i][0])
            class_metric[i]["recall"].append(metrics[i][1])
            class_metric[i]["f1_score"].append(metrics[i][2])
            acc_precision += class_metric[i]["precision"][-1]
            acc_recall += class_metric[i]["recall"][-1]
            acc_f1_score += class_metric[i]["f1_score"][-1]
        # Append averages of each metric to lists
        avg_precision_list.append(acc_precision / n_classes)
        avg_recall_list.append(acc_recall / n_classes)
        avg_f1_score_list.append(acc_f1_score / n_classes)
        update_steps.append(update_step)

    # Plot average precision over time
    plt.plot(update_steps, avg_precision_list)
    plt.title("Average precision over time")
    plt.xlabel("Update step")
    plt.ylabel("Average precision")
    plt.savefig(os.path.join(save_dir, 'avg_precision.png'))
    plt.cla()
    # Plot average recall over time
    plt.plot(update_steps, avg_recall_list)
    plt.title("Average recall over time")
    plt.xlabel("Update step")
    plt.ylabel("Average recall")
    plt.savefig(os.path.join(save_dir, 'avg_recall.png'))
    plt.cla()
    # Plot average f-1 score over time
    plt.plot(update_steps, avg_f1_score_list)
    plt.title("Average f-1 score over time")
    plt.xlabel("Update step")
    plt.ylabel("Average f-1 score")
    plt.savefig(os.path.join(save_dir, 'avg_f1.png'))
    plt.cla()

    # For each class, plot precision recall and f-1 score over time
    for i in range(n_classes):
        plt.plot(update_steps, class_metric[i]["precision"])
        # Plot class precision
        plt.title("Precision over time: {}".format(label_dict[i]))
        plt.xlabel("Update step")
        plt.ylabel("Average precision")
        plt.savefig(os.path.join(save_dir, '{}-precision.png'.format(i)))
        plt.cla()
        # Plot class recall
        plt.plot(update_steps, class_metric[i]["recall"])
        plt.title("Recall over time: {}".format(label_dict[i]))
        plt.xlabel("Update step")
        plt.ylabel("Average recall")
        plt.savefig(os.path.join(save_dir, '{}-recall.png'.format(i)))
        plt.cla()
        # Plot class f-1 score
        plt.plot(update_steps, class_metric[i]["f1_score"])
        plt.title("F-1 score over time: {}".format(label_dict[i]))
        plt.xlabel("Update step")
        plt.ylabel("Average f-1 score")
        plt.savefig(os.path.join(save_dir, '{}.f1.png'.format(i)))
        plt.cla()

    # Plot encoder output norm over time
    plt.plot(update_steps, avg_dataset_norms)
    plt.title("Average encoder's output norm over time")
    plt.xlabel("Update step")
    plt.ylabel("Average encoder output norm")
    plt.savefig(os.path.join(save_dir, 'encoder_output_norm.png'))
    plt.cla()


def save(results, save_dir):
    """ Saves the evaluation results to an npz file.

    Args:
        results: list of tuples where each tuple is the update step number along with a numpy array containing
            the precision, recall and f1-score for each class.
        save_dir: The directory where the results should be saved.

    Returns:
        None
    """
    array_dict = {}
    for update_step, result in results:
        array_dict[str(update_step)] = result
    np.savez(os.path.join(save_dir, "evaluation_over_time_results.npz"), **array_dict)


if __name__ == "__main__":
    args = get_arguments()
    tfrecord_path_train = args.tfrecord_path_train
    tfrecord_path_test = args.tfrecord_path_test
    checkpoints_dir = args.checkpoints_dir
    save_dir = args.save_dir
    evaluation_model = args.evaluation_model
    name = args.model_name

    print("-Getting and sorting checkpoint paths according to update step")
    checkpoint_tuples = get_checkpoint_paths(checkpoints_dir, name=name)

    print("-Read labels and actor meta-data for training dataset samples.")
    y_train, actors_train = load_data_labels(tfrecord_path_train)
    label_dict = {value: index for index, value in enumerate(np.unique(y_train))}
    actor_dict = {value: index for index, value in enumerate(np.unique(actors_train))}
    y_train = np.vectorize(label_dict.get)(y_train)
    actors = np.vectorize(actor_dict.get)(actors_train)
    unique_actors_train = np.unique(actors_train)

    rint("-Calculating class_weights based on the training data class labels")
    class_counts = [len(y_train[y_train == i]) for i in label_dict.values()]
    class_weights = [max(class_counts) / class_count for class_count in class_counts]
    class_weight_dict = {class_idx: class_weight for class_idx, class_weight in zip(label_dict.values(), class_weights)}

    print("-Read labels and actor meta-data for test dataset samples.")
    y_test, actors_test = load_data_labels(tfrecord_path_test)
    label_dict = {value: index for index, value in enumerate(np.unique(y_test))}
    actor_dict = {value: index for index, value in enumerate(np.unique(actors_test))}
    y_test = np.vectorize(label_dict.get)(y_test)
    actors_test = np.vectorize(actor_dict.get)(actors_test)
    unique_actors_test = np.unique(actors_test)

    train_sample_weight = [class_weight_dict[label] for label in y_train]
    test_sample_weight = [class_weight_dict[label] for label in y_test]

    print("-Start evaluating features for each checkpoint")
    results = []
    avg_dataset_norms = []
    FNULL = open(os.devnull, 'w')
    random_feature_filename = str(random.getrandbits(64)) + ".npy"

    for update_step, checkpoint_path in tqdm(checkpoint_tuples):
        tqdm.write("Extracting features with model from checkpoint: {}".format(checkpoint_path))
        # Run feature extraction script for the training set - removing the print outs
        subprocess.call(" python -m feature_extraction.extract_features --tfrecord-path={} --checkpoint-step={} "
                        "--features-path={}".format(tfrecord_path_train, update_step, random_feature_filename),
                        shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        # Load features
        X_train = np.load(random_feature_filename)

        # Run feature extraction script for the test set - removing the print outs
        subprocess.call(" python -m feature_extraction.extract_features --tfrecord-path={} --checkpoint-step={} "
                        "--features-path={}".format(tfrecord_path_test, update_step, random_feature_filename),
                        shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
        # Load features
        X_test = np.load(random_feature_filename)

        tqdm.write("Evaluating features extracted with model from checkpoint {}.".format(checkpoint_path))

        # Get norm of encoder's output normalizing the features
        dataset_norms = np.linalg.norm(X_train, axis=0)
        avg_dataset_norms.append(np.mean(dataset_norms))

        # Normalize features
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Train model
        if evaluation_model == LOGISTIC_REGRESSION_STR:
            model = LogisticRegression(multi_class='multinomial', random_state=0)
            model = model.fit(X_train, y_train, sample_weight=train_sample_weight)
        elif evaluation_model == RANDOM_FOREST_STR:
            model = RandomForestClassifier(n_estimators=300, random_state=0)
            model = model.fit(X_train, y_train, sample_weight=train_sample_weight)

        # Evaluate model
        y_preds = model.predict(X_test)
        metrics = np.array(precision_recall_fscore_support(y_test, y_preds)[:3]).T  # Omit supports

        results.append((update_step, metrics))
    os.remove(random_feature_filename)

    print("-Visualizing and saving results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inv_label_dict = {v: k for k, v in label_dict.items()}
    visualize(results, inv_label_dict, save_dir, avg_dataset_norms)
    save(results, save_dir)




















