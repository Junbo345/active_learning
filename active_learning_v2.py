import logging
import math
from warnings import simplefilter
from dataclasses import dataclass, field

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import matplotlib.ticker as ticker

from sklearn.cluster import KMeans, kmeans_plusplus

import estimators_v2 as estimators


# Default configuration values using OmegaConf and dataclasses

@dataclass
class LoopConfig:
    times: int = 50
    init: int = 50
    batch_size: int = 15


@dataclass
class Learner:
    regression_method: str = 'L'


@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])
    label_cols: list = field(default_factory=lambda: ['s1_lrg_fraction', 's1_spiral_fraction', 'other'])


def label_highest_prob(row):
    """
    Convert a row of probabilities to a one-hot encoded vector where the highest probability is set to 1.
    """
    max_value_index = row.argmax()  # Find the index of the maximum value in the row
    new_row = [0] * len(row)  # Set all values to 0
    new_row[max_value_index] = 1  # Set the index of the maximum value to 1
    return new_row


def loaddata(cfg) -> pd.DataFrame:
    """
    Load the galaxy dataset, preprocess it, and return a DataFrame with features and one-hot encoded labels.
    """

    expected_csv_loc = 'galaxy.csv'  # Expect the CSV file to be in the current folder

    df = pd.read_csv(expected_csv_loc)
    df['other'] = 1 - (df['s1_lrg_fraction'] + df['s1_spiral_fraction'])

    # Load the data and select specified columns
    columns_to_select = cfg.feature_cols + cfg.label_cols
    df_s = df[columns_to_select]

    # Apply the label_highest_prob function to each row and convert to list of integers
    label_columns_expanded = df_s[cfg.label_cols].apply(label_highest_prob, axis=1, result_type='expand')

    # Ensure the expanded label columns have the same names as the original label columns
    label_columns_expanded.columns = cfg.label_cols

    # Concatenate the feature columns with the expanded label columns
    df_final = pd.concat([df_s[cfg.feature_cols], label_columns_expanded], axis=1)

    return df_final


def get_data(iterations, initial, batch, method):
    """
    Generate data for plotting the performance of different sampling methods over iterations.

    Parameters:
    iterations (List[int]): Number of iterations for each method
    initial (List[int]): Initial pool of data for each method
    batch (List[int]): Number of data points in each batch of iteration
    method (List[str]): Regression method for each iteration
    """
    col = max(iterations)

    cfg = OmegaConf.structured(ActiveLearningConfig)
    test = loaddata(cfg).sample(n=1500, random_state=1)
    train = loaddata(cfg).drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)
    output = pd.DataFrame()

    for ind in range(len(batch)):
        cfg.loop.times = iterations[ind]
        cfg.loop.init = initial[ind]
        cfg.loop.batch_size = batch[ind]
        cfg.learner.regression_method = method[ind]

        # for a quick baseline check
        # print(get_y_baseline(train, test, cfg))
        # continue

        # Initialize the X-axis values for the plot
        # this is now number of labelled points (galaxies)
        x = np.arange(cfg.loop.times) * cfg.loop.batch_size + cfg.loop.init

        # Calculate the Y-axis values for different sampling methods

        y1 = abs(np.log(1 - np.array(badge(train, test, cfg))))
        y2 = abs(np.log(1 - np.array(ambiguous(train, test, cfg))))
        # y3 = abs(np.log(1 - np.array(diverse_tree(train, test, cfg))))
        output[f'x{ind + 1} {method[ind]}'] = np.concatenate([x, np.full(col - len(x), np.nan)])
        output[f'badge_{ind + 1}'] = np.concatenate([y1, np.full(col - len(x), np.nan)])

        output[f'uncertainty_{ind + 1}'] = np.concatenate([y2, np.full(col - len(x), np.nan)])
        # output[f'div_{ind + 1}'] = np.concatenate([y3, np.full(col - len(x), np.nan)])

    # Save the output to a CSV file

    csv_file_path = r'C:/Users/20199/Desktop/model.csv'

    output.to_csv(csv_file_path, index=False)


def get_y_baseline(train, test, cfg):
    """
    Compute the baseline score using the specified regression method.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    float: Baseline score
    """
    X = train[cfg.feature_cols].values
    y = train[cfg.label_cols].values

    X_test = test[cfg.feature_cols].values
    y_test = test[cfg.label_cols].values
    clf = get_active_learner(cfg)
    y_baseline = abs(math.log(1 - clf.fit(X, y).score(X_test, y_test)))
    return y_baseline


def get_active_learner(cfg):
    """
    Initialize and return an active learner based on the specified regression method in the config.

    Parameters:
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    estimator: Initialized estimator
    """
    if cfg.learner.regression_method == "L":
        clf = LogisticRegression()
    elif cfg.learner.regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)
    elif cfg.learner.regression_method == "pytorch_N":
        model = estimators.MLP(20, 3)  # 20 features, 3 classes
        clf = estimators.LightningEstimator(model, max_epochs=1000, batch_size=500)
    elif cfg.learner.regression_method == "pytorch_L":
        model = estimators.Multiregression(20, 3)  # 20 features, 3 classes
        clf = estimators.LightningEstimator(model, max_epochs=1000, batch_size=500)
    else:
        raise ValueError(f"Invalid regression method: {cfg.learner.regression_method}")
    return clf


def randomapp(train, test, cfg):
    """
    Evaluate the performance of random sampling over multiple iterations.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    List[float]: Test scores for each iteration
    """
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    clf = get_active_learner(cfg)
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index)
    trace = []

    for iteration_n in range(cfg.loop.times):
        new = pool.sample(n=cfg.loop.batch_size)
        labelled = pd.concat([labelled, new])
        pool = pool.drop(new.index)
        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values
        test_x = test[cfg.feature_cols].values
        test_y = test[cfg.label_cols].values
        clf.fit(X, y)
        trace.append(clf.score(test_x, test_y))
    return trace


def ambiguous(train, test, cfg):
    """
    Evaluate the performance of ambiguous sampling over multiple iterations.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    List[float]: Test scores for each iteration
    """
    clf = get_active_learner(cfg)
    label = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    for i in range(cfg.loop.times):

        X_label = label[cfg.feature_cols].values
        y_label = label[cfg.label_cols].values
        model = clf.fit(X_label, y_label)
        X_pool = pool[cfg.feature_cols].values

        pred_proba = model.predict_proba(X_pool)[:, 1]
        diff = np.abs(0.333 - pred_proba)
        top_n_indices = np.argsort(diff)[:cfg.loop.batch_size]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(pool.index[top_n_indices]).reset_index(drop=True)

        test_x = test[cfg.feature_cols].values
        test_y = test[cfg.label_cols].values
        score = model.score(test_x, test_y)

        number.append(score)

    return number


def diverse_matrix(train, test, cfg):
    """
    Evaluate the performance of diverse sampling using a distance-based approach over multiple iterations.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    List[float]: Test scores for each iteration
    """
    clf = get_active_learner(cfg)
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index).reset_index(drop=True)
    trace = []

    for iteration_n in range(cfg.loop.times):
        X_labelled = labelled[cfg.feature_cols].values
        y_labelled = labelled[cfg.label_cols].values
        clf.fit(X_labelled, y_labelled)
        X_pool = pool[cfg.feature_cols].values
        distances = cdist(X_labelled, X_pool)
        mean_distances = distances.mean(axis=0)
        indices_to_add = mean_distances.argsort()[:cfg.loop.batch_size]
        labelled = pd.concat([labelled, pool.iloc[indices_to_add]], ignore_index=True)
        pool = pool.drop(pool.index[indices_to_add]).reset_index(drop=True)
        X_test = test[cfg.feature_cols].values
        y_test = test[cfg.label_cols].values
        score = clf.score(X_test, y_test)
        trace.append(score)

    return trace


def diverse_tree(train, test, cfg):
    """
    Evaluate the performance of diverse sampling using a KD-Tree approach over multiple iterations.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    List[float]: Test scores for each iteration
    """
    clf = get_active_learner(cfg)
    label = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    for i in range(cfg.loop.times):
        X_label = label.iloc[:, :-3].values
        y_label = label.iloc[:, -3:].values
        model = clf.fit(X_label, y_label)
        X_pool = pool.iloc[:, :-3].values
        tree = cKDTree(X_label)
        distances, indices = tree.query(X_pool, k=1)
        top_n_indices = np.argsort(distances)[-cfg.loop.batch_size:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(pool.index[top_n_indices]).reset_index(drop=True)
        X_test = test.iloc[:, :-3].values
        y_test = test.iloc[:, -3:].values
        score = model.score(X_test, y_test)
        number.append(score)

    return number


def find_row_indices(subset_array, target_array):
    indices = []
    for subset_row in subset_array:
        # Find the index of the matching row in the target array
        for i, target_row in enumerate(target_array):
            if np.array_equal(subset_row, target_row):
                indices.append(i)
                break
    return indices

def badge(train, test, cfg):
    """
    Evaluate the performance of random sampling over multiple iterations.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    cfg (ActiveLearningConfig): Configuration for active learning

    Returns:
    List[float]: Test scores for each iteration
    """
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    clf = get_active_learner(cfg)
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index).reset_index(drop=True)
    trace = []

    for k in range(cfg.loop.times):

        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values
        clf.fit(X, y)
        X_quary = pool[cfg.feature_cols].values
        y_result = clf.predict_proba(X_quary)
        predicted_classes = np.argmax(y_result, axis=1)
        y_onehot = np.zeros_like(y_result)
        y_onehot[np.arange(y_result.shape[0]), predicted_classes] = 1

        y_final = y_result - y_onehot

        X_tensor = torch.tensor(X_quary, dtype=torch.float32)
        with torch.no_grad():
            key = torch.relu(clf.model.linear1(X_tensor)).numpy()

        g_x_mul = []

        for _ in range(len(key)):
            num_cols_array1 = y_final.shape[1]  # 3
            num_cols_array2 = key.shape[1]
            g_x_ind = np.empty((num_cols_array1, num_cols_array2))
            for n_col in range(y_result.shape[1]):
                g_x_ind[n_col, :] = y_final[_, n_col] * key[_, :]
            g_x_mul.append(g_x_ind.flatten())

        g_x_final = np.array(g_x_mul)
        kmeans, indices = kmeans_plusplus(g_x_final,n_clusters=cfg.loop.batch_size, random_state=0)


        rows_to_transfer = pool.loc[indices]
        labelled = pd.concat([labelled, rows_to_transfer], ignore_index=True)
        pool = pool.drop(indices).reset_index(drop=True)
        test_x = test[cfg.feature_cols].values
        test_y = test[cfg.label_cols].values
        trace.append(clf.score(test_x, test_y))

    return trace


def plot():
    """
    Generate and display plots to visualize the performance of different sampling methods.
    """
    data = pd.read_csv(r'C:/Users/20199/Desktop/model.csv')
    labels = list(data.columns)
    x_labels = [l for l in labels if l.startswith("x")]
    y_labels = [l for l in labels if not l.startswith("x")]
    methods = list(set(label.split()[1] for label in x_labels))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    axs = axs.flatten()
    colors = ["red", "blue", "green"]

    for i, method in enumerate(methods):
        ax = axs[i]
        for j, color in enumerate(colors):
            ax.plot(data[f'x{j + 1} {method}'], data[f'{["ran", "amb", "div"][j]}_{j + 1}'], color=color,
                    label=f'{["Random", "Ambiguous", "Diverse"][j]} Sampling')
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Score")
        ax.set_title(f'Sampling Method: {method}')
        ax.legend()
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # https://lightning.ai/docs/pytorch/stable/extensions/logging.html#configure-console-logging
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    # ignore all warnings
    simplefilter(action='ignore')

    # loaddata()


    get_data(iterations=[30], initial=[50], batch=[100], method=["pytorch_N"])

    # get_data(iterations=[75], initial=[50], batch=[30], method=["pytorch_N"])

