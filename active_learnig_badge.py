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
import estimators_badge as estimators
from sklearn.cluster import kmeans_plusplus


# Default configuration values using OmegaConf and dataclasses

@dataclass
class LoopConfig:
    times: int = 50  # Number of iterations in the active learning loop
    init: int = 50  # Initial number of samples to start the active learning process
    batch_size: int = 15  # Number of samples to add in each iteration


@dataclass
class Learner:
    regression_method: str = 'L'  # Default regression method (L: Logistic Regression, M: MLPClassifier)


@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])  # Feature column names
    label_cols: list = field(default_factory=lambda: ['s1_lrg_fraction', 's1_spiral_fraction', 's1_other_fraction'])  # Label column names

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
    # df['other'] = 1 - (df['s1_lrg_fraction'] + df['s1_spiral_fraction'])

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
    iterations (list[int]): Number of iterations for each method
    initial (list[int]): Initial pool of data for each method
    batch (list[int]): Number of data points in each batch of iteration
    method (list[str]): Regression method for each iteration
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
        x = np.arange(cfg.loop.times+1) * cfg.loop.batch_size + cfg.loop.init

        # Calculate the Y-axis values for different sampling methods
        y1 = abs(np.log(1 - np.array(badge(train, test, cfg))))
        y2 = abs(np.log(1 - np.array(diverse_tree(train, test, cfg))))
        y3 = abs(np.log(1 - np.array(randomapp(train, test, cfg))))
        output[f'x{ind + 1} {method[ind]}'] = np.concatenate([x, np.full(col - len(x)+1, np.nan)])
        output[f'badge_{ind + 1}'] = np.concatenate([y1, np.full(col - len(x)+1, np.nan)])
        output[f'uncertainty_{ind + 1}'] = np.concatenate([y2, np.full(col - len(x)+1, np.nan)])
        output[f'random_{ind + 1}'] = np.concatenate([y3, np.full(col - len(x)+1, np.nan)])

    # Save the output to a CSV file
    csv_file_path = # Add path
    output.to_csv(csv_file_path, index=False)


def get_y_baseline(train, test, cfg):
    """
    Compute the baseline score using the specified regression method. It is done by fitting all the data in the training model to the model.

    Parameters:
    train (DataFrame): Data for training. Here, it should contain columns of input features and columns of output labels (in one-hot encoding).
    test (DataFrame): Data for testing, same structure as training data
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
    Initialize and return an active learner based on the specified regression method in the config. 'L' and 'M' are logistic regression and MLP implemented by scikit-learn. 'pytorch_N' and 'pytorch_M' are logistic regression and MLP implemented through
    pytorch. If using Badge, 'pytorch_N' and 'pytorch_M' should be chosen.

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
        model = estimators.MLP(20, [128],3)  # 20 features, 3 classes
        clf = estimators.LightningEstimator(model, max_epochs=1000, batch_size=500)
    elif cfg.learner.regression_method == "pytorch_L":
        model = estimators.Multiregression(20, 3)  # 20 features, 3 classes
        clf = estimators.LightningEstimator(model, max_epochs=1000, batch_size=500)
    else:
        raise ValueError(f"Invalid regression method: {cfg.learner.regression_method}")
    return clf


def randomapp(train, test, cfg):
    """
    Evaluate the performance of random appending on specified ML models. Sampling over multiple iterations.

    Parameters:
    train (DataFrame): Data for training. Should contain columns of input features and columns of output labels (in one-hot encoding).
    test (DataFrame): Data for testing, same structure as training data
    cfg (ActiveLearningConfig): Configuration for active learning, which includes initial sampling size,
                                number of iterations, feature columns, label columns, and batch size.

    Returns:
    List[float]: Test scores for each iteration
    """
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    clf = get_active_learner(cfg)
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index)
    test_x = test[cfg.feature_cols].values
    test_y = test[cfg.label_cols].values
    trace = []
    X = labelled[cfg.feature_cols].values
    y = labelled[cfg.label_cols].values
    clf.fit(X, y)
    trace.append(clf.score(test_x, test_y))


    for iteration_n in range(cfg.loop.times):
        new = pool.sample(n=cfg.loop.batch_size)
        labelled = pd.concat([labelled, new])
        pool = pool.drop(new.index)
        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values
        clf.fit(X, y)
        trace.append(clf.score(test_x, test_y))
    return trace


def ambiguous(train, test, cfg):
    """
    Evaluate the performance of ambiguous active learning algorithms (highest probs closest to 1/n, n = # of classes) on specified ML models. 
    Sampling over multiple iterations.
    
    Parameters:
    train (DataFrame): Training data, see randomapp for details
    test (DataFrame): Testing data, see randomapp for details
    cfg (ActiveLearningConfig): Configuration for active learning, see randomapp for details

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

    X_label = label[cfg.feature_cols].values
    y_label = label[cfg.label_cols].values
    model = clf.fit(X_label, y_label)
    number.append(model.score(test_x, test_y))
    return number


def diverse_matrix(train, test, cfg):
    """
    Evaluate the performance of diverse active learning algorithms (most distant points wrt points already been labeled) on specified ML models. 
    Sampling over multiple iterations.

    Parameters:
    train (DataFrame): Training data, see randomapp for details
    test (DataFrame): Testing data, see randomapp for details
    cfg (ActiveLearningConfig): Configuration for active learning, see randomapp for details

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
    Evaluate the performance of diverse active learning algorithms (most distant points wrt points already been labeled) on specified ML models. 
    Sampling over multiple iterations.
    Performs same as dicerse_matrix. But used CDK-Tree for distance calculation, should run faster. 

    Parameters:
    train (DataFrame): Training data, see randomapp for details
    test (DataFrame): Testing data, see randomapp for details
    cfg (ActiveLearningConfig): Configuration for active learning, see randomapp for details

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
        tree = cKDTree(X_label)
        distances, indices = tree.query(X_pool, k=1)
        top_n_indices = np.argsort(distances)[-cfg.loop.batch_size:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(pool.index[top_n_indices]).reset_index(drop=True)
        X_test = test[cfg.feature_cols].values
        y_test = test[cfg.label_cols].values
        score = model.score(X_test, y_test)
        number.append(score)

    X_label = label[cfg.feature_cols].values
    y_label = label[cfg.label_cols].values
    model = clf.fit(X_label, y_label)
    number.append(model.score(test[cfg.feature_cols].values, test[cfg.label_cols].values))

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
    Evaluate the performance of BADGE active learning algorithms (most distant points wrt points already been labeled) on specified ML models. 
    https://arxiv.org/pdf/1906.03671
    Sampling over multiple iterations. 

    Parameters:
    train (DataFrame): Training data, see randomapp for details
    test (DataFrame): Testing data, see randomapp for details
    cfg (ActiveLearningConfig): Configuration for active learning, see randomapp for details

    Returns:
    List[float]: Test scores for each iteration of active learning.
    """
    # Validate input data
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Initialize the active learner
    clf = get_active_learner(cfg)

    # Initial sampling from the training data
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index).reset_index(drop=True)

    # List to store test scores for each iteration
    trace = []

    # Active learning loop
    for k in range(cfg.loop.times):
        # Extract features and labels for the current labelled data
        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values

        # Train the model on the current labelled data
        clf.fit(X, y)

        # Predict probabilities on the unlabeled pool
        X_query = pool[cfg.feature_cols].values
        y_result = clf.predict_proba(X_query)

        # Get predicted classes and create one-hot encoding
        predicted_classes = np.argmax(y_result, axis=1)
        y_onehot = np.zeros_like(y_result)
        y_onehot[np.arange(y_result.shape[0]), predicted_classes] = 1

        # Compute the difference between probabilities and one-hot encodings
        y_final = y_result - y_onehot

        # Extract model's second last layer output (embeddings)
        key = clf.model.second_last_output.detach().numpy()

        # Compute gradient embeddings for BADGE
        g_x_mul = []
        for _ in range(len(key)):
            num_cols_array1 = y_final.shape[1]
            num_cols_array2 = key.shape[1]
            g_x_ind = np.empty((num_cols_array1, num_cols_array2))
            for n_col in range(y_result.shape[1]):
                g_x_ind[n_col, :] = y_final[_, n_col] * key[_, :]
            g_x_mul.append(g_x_ind.flatten())

        g_x_final = np.array(g_x_mul)

        # Apply k-means++ to select diverse samples for the next batch
        kmeans, indices = kmeans_plusplus(g_x_final, n_clusters=cfg.loop.batch_size, random_state=0)

        # Add the selected samples to the labelled set and remove from the pool
        rows_to_transfer = pool.loc[indices]
        labelled = pd.concat([labelled, rows_to_transfer], ignore_index=True)
        pool = pool.drop(indices).reset_index(drop=True)

        # Evaluate the model on the test data and store the score
        test_x = test[cfg.feature_cols].values
        test_y = test[cfg.label_cols].values
        trace.append(clf.score(test_x, test_y))

    X_label = labelled[cfg.feature_cols].values
    y_label = labelled[cfg.label_cols].values
    model = clf.fit(X_label, y_label)
    trace.append(model.score(test_x, test_y))

    return trace

