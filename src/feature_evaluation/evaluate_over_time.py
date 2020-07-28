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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_NAME = "wgan"
DEFAULT_SAVE_DIR = os.path.join("..", "results")

# Dataset sub-directory names
INPUT_DATA_DIR = "input_data"
LABELS_DIR = "labels"

# Define strings that denote the different sk-learn models to be used for feature evaluation
LOGISTIC_REGRESSION_STR = "LogisticRegression"
RANDOM_FOREST_STR = "RandomForest"

CHANCE_LEVEL_METRICS = {
    "precision": 0.1,
    "recall": 0.1,
    "f1-score": 0.1,
    "accuracy": 0.1
}


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--train-path", type=str, required=True,
                        help="Path to directory where the training dataset is stored.")
    parser.add_argument("--test-path", type=str, required=True,
                        help="Path to directory where the test dataset is stored.")
    parser.add_argument("--checkpoints-dir", type=str, required=True,
                        help="Path to the directory with all the checkpoints for the model.")
    parser.add_argument("--save-dir", type=str, required=False, default=DEFAULT_SAVE_DIR,
                        help="Path to the directory where results should be saved.")
    parser.add_argument("--evaluation-model", type=str, required=False, default=LOGISTIC_REGRESSION_STR,
                        help="String denoting the type of sklearn model to be used for evaluating the features.")
    parser.add_argument("--model-name", type=str, required=False, default=DEFAULT_NAME,
                        help="The name of the trained model included in the checkpoint file names.")
    return parser.parse_args()


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
        results: A list of tuples where each tuple contains the update step number (1st element),
            along with a numpy array (2nd element) containing the precision, recall and f1-score
            for each class, and finally the accuracy (3rd element).
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
    accuracy_list = []
    update_steps = []
    for update_step, metrics, accuracy in results:
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
        accuracy_list.append(accuracy)
        update_steps.append(update_step)

    # Plot average precision over time
    plt.plot(update_steps, avg_precision_list)
    plt.plot(update_steps, [CHANCE_LEVEL_METRICS["precision"] for i in range(len(update_steps))], '--')
    plt.title("Average precision over time")
    plt.xlabel("Update step")
    plt.ylabel("Average precision")
    plt.savefig(os.path.join(save_dir, 'avg_precision.png'))
    plt.cla()
    # Plot average recall over time
    plt.plot(update_steps, avg_recall_list)
    plt.plot(update_steps, [CHANCE_LEVEL_METRICS["recall"] for i in range(len(update_steps))], '--')
    plt.title("Average recall over time")
    plt.xlabel("Update step")
    plt.ylabel("Average recall")
    plt.savefig(os.path.join(save_dir, 'avg_recall.png'))
    plt.cla()
    # Plot average f-1 score over time
    plt.plot(update_steps, avg_f1_score_list)
    plt.plot(update_steps, [CHANCE_LEVEL_METRICS["f1-score"] for i in range(len(update_steps))], '--')
    plt.title("Average f-1 score over time")
    plt.xlabel("Update step")
    plt.ylabel("Average f-1 score")
    plt.savefig(os.path.join(save_dir, 'avg_f1.png'))
    plt.cla()
    # Plot accuracy over time
    plt.plot(update_steps, accuracy_list)
    plt.plot(update_steps, [CHANCE_LEVEL_METRICS["accuracy"] for i in range(len(update_steps))], '--')
    plt.title("Accuracy over time")
    plt.xlabel("Update step")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
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
    acc_dict = {}
    for update_step, result, accuracy in results:
        array_dict[str(update_step)] = result
        acc_dict[str(update_step)] = accuracy
    np.savez(os.path.join(save_dir, "evaluation_over_time_results.npz"), **array_dict)
    np.savez(os.path.join(save_dir, "evaluation_over_time_accuracies.npz"), **array_dict)


if __name__ == "__main__":
    args = get_arguments()
    train_path = args.train_path
    test_path = args.test_path
    checkpoints_dir = args.checkpoints_dir
    save_dir = args.save_dir
    evaluation_model = args.evaluation_model
    name = args.model_name

    training_input_path = os.path.join(train_path, INPUT_DATA_DIR)
    training_labels_path = os.path.join(train_path, LABELS_DIR)

    test_input_path = os.path.join(test_path, INPUT_DATA_DIR)
    test_labels_path = os.path.join(test_path, LABELS_DIR)

    print("-Getting and sorting checkpoint paths according to update step")
    checkpoint_tuples = get_checkpoint_paths(checkpoints_dir, name=name)

    print("-Read label meta-data for training dataset samples.")
    y_train = load_data_labels(training_labels_path)
    label_dict = {value: index for index, value in enumerate(np.unique(y_train))}
    y_train = np.vectorize(label_dict.get)(y_train)

    print("-Calculating class_weights based on the training data class labels")
    class_counts = [len(y_train[y_train == i]) for i in label_dict.values()]
    class_weights = [max(class_counts) / class_count for class_count in class_counts]
    class_weight_dict = {class_idx: class_weight for class_idx, class_weight in zip(label_dict.values(), class_weights)}

    print("-Read label meta-data for test dataset samples.")
    y_test = load_data_labels(test_labels_path)
    label_dict = {value: index for index, value in enumerate(np.unique(y_test))}
    y_test = np.vectorize(label_dict.get)(y_test)

    train_sample_weight = [class_weight_dict[label] for label in y_train]
    test_sample_weight = [class_weight_dict[label] for label in y_test]

    print("-Start evaluating features for each checkpoint")
    results = []
    avg_dataset_norms = []
    FNULL = open(os.devnull, 'w')
    random_feature_filename = str(random.getrandbits(64)) + ".npy"

    for update_step, checkpoint_path in tqdm(checkpoint_tuples):
        # Get the parent directory to the checkpoint directory
        results_path = checkpoint_path.rstrip(os.path.sep)
        results_path = os.path.join(checkpoint_path, "..", "..")

        tqdm.write("Extracting features with model from checkpoint: {}".format(checkpoint_path))
        # Run feature extraction script for the training set - removing the print outs
        subprocess.call(" python -m feature_extraction.extract_features --dataset-path={} --checkpoint-step={} "
                        "--features-path={} --results-dir={}"
                        .format(training_input_path, update_step, random_feature_filename, results_path),
                        shell=True)
        # Load features
        X_train = np.load(random_feature_filename)

        # Run feature extraction script for the test set - removing the print outs
        subprocess.call(" python -m feature_extraction.extract_features --dataset-path={} --checkpoint-step={} "
                        "--features-path={} --results-dir={}"
                        .format(test_input_path, update_step, random_feature_filename, results_path),
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
        accuracy = accuracy_score(y_test, y_preds)

        results.append((update_step, metrics, accuracy))
    os.remove(random_feature_filename)

    print("-Visualizing and saving results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    inv_label_dict = {v: k for k, v in label_dict.items()}
    visualize(results, inv_label_dict, save_dir, avg_dataset_norms)
    save(results, save_dir)




















