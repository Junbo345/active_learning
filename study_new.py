import logging
import math
from warnings import simplefilter
from dataclasses import dataclass, field

from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

import test as estimators

# default config values

# https://omegaconf.readthedocs.io/en/2.1_branch/structured_config.html#nesting-structured-configs
@dataclass
class LoopConfig:
    times: int=15
    init: int=50
    batch_size: int=15

@dataclass
class Learner:
    regression_method: str='L'

@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])
    label_cols: list = field(default_factory=lambda: ['s1_lrg_fraction', 's1_spiral_fraction', 'other'])


def label_highest_prob(row):
    # Find the index of the maximum value in the row
    max_value_index = row.argmax()
    # Set all values to 0
    new_row = [0] * len(row)
    # Set the index of the maximum value to 1
    new_row[max_value_index] = 1
    return new_row


def loaddata(cfg) -> pd.DataFrame:
    # Expect the CSV file to be in the current folder
    expected_csv_loc = 'galaxy.csv'
    df = pd.read_csv(expected_csv_loc)
    df['other'] = 1 - (df['s1_lrg_fraction'] + df['s1_spiral_fraction'])

    # Load the data
    columns_to_select = cfg.feature_cols + cfg.label_cols
    df_s = df[columns_to_select]

    # Apply the label_highest_prob function to each row and convert to list of integers
    label_columns_expanded = df_s[cfg.label_cols].apply(label_highest_prob, axis=1, result_type='expand')

    # Ensure the expanded label columns have the same names as the original label columns
    label_columns_expanded.columns = cfg.label_cols

    # Concatenate the feature columns with the expanded label columns
    df_final = pd.concat([df_s[cfg.feature_cols], label_columns_expanded], axis=1)

    return df_final


def plot():
    """
    Function to plot the performance of different sampling methods over iterations.

    Parameters:
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration (default = 1)
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)
    """
    # keep all hparams here
    cfg = OmegaConf.structured(ActiveLearningConfig)
    # can override config values here

    cfg.learner.regression_method = 'pytorch'
    # cfg.learner.regression_method = 'L'

    print(OmegaConf.to_yaml(cfg))

    # Load and split the data into test and train sets
    test = loaddata(cfg).sample(n=2000, random_state=1)
    train = loaddata(cfg).drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)



    # Initialize the X-axis values for the plot
    x = np.arange(cfg.loop.times)

    # Calculate the Y-axis values for different sampling methods
    y1 = abs(np.log(1 - np.array(randomapp(train, test, cfg))))
    y2 = abs(np.log(1 - np.array(ambiguous(train, test, cfg))))
    y3 = abs(np.log(1 - np.array(diverse_tree(train, test, cfg))))
    # y_baseline = get_y_baseline(train, test, cfg)

    # Plot a horizontal line representing the baseline score
    # plt.axhline(y=y_baseline, color='r', linestyle='-')

    # Plot the performance of the random sampling method
    plt.plot(x, y1, label='random', marker='o')

    # Plot the performance of the ambiguous sampling method
    plt.plot(x, y2, label='ambiguous', marker='s')

    # Plot the performance of the diverse sampling method
    plt.plot(x, y3, label='diverse', marker='s')

    # Set the labels for the X and Y axes
    plt.xlabel('Iterations')
    plt.ylabel('Score')

    # Set the title of the plot, including the parameters
    title = (f'Three Lines with Different Methods\nIterations: {cfg.loop.times}, Initial sample size: {cfg.loop.init}, '
             f'Batch size: {cfg.loop.batch_size}, Regression Method: {cfg.learner.regression_method}')
    plt.title(title)

    # Display the legend
    plt.legend()

    # Show the plot
    return plt.show()


def get_y_baseline(train, test, cfg):
    X = train.iloc[:, :(train.shape[1] - 1)].values
    y = train.iloc[:, -1].values

    X_test = test.iloc[:, :(test.shape[1] - 1)].values
    y_test = test.iloc[:, -1].values
    clf = get_active_learner(cfg)
    y_baseline = abs(math.log(1 - clf.fit(X, y).score(X_test,y_test)))
    return y_baseline

def get_active_learner(cfg):
    if cfg.learner.regression_method == "L":
        clf = LogisticRegression()
    elif cfg.learner.regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)
    elif cfg.learner.regression_method == "pytorch":
        model = estimators.Multiregression(20, 3)  # 20 features, 2 classes
        clf = estimators.LightningEstimator(model, max_epochs=10, batch_size=500)
    else:
        raise ValueError(f"Invalid regression method: {cfg.learner.regression_method}")
    return clf


def randomapp(train, test, cfg):
    """
    Function to return test scores of random append.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)

    Returns:
    List: Test scores for each iteration
    """

    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    clf = get_active_learner(cfg)

    # Initialize the label with a sample from the train data
    labelled = train.sample(n=cfg.loop.init)

    # Remove the sampled label data from the pool
    pool = train.drop(labelled.index)

    # Initialize a list to store test scores for each iteration
    trace = []

    # Iteratively select data points and evaluate the model
    for iteration_n in range(cfg.loop.times):
        print('Iteration: ', iteration_n)
        # Randomly sample data points from the pool
        new = pool.sample(n=cfg.loop.batch_size)

        # Add the new samples to the labelled
        labelled = pd.concat([labelled, new])
        print(len(labelled))

        # Remove the new samples from the pool
        pool = pool.drop(new.index)

        # Prepare data for model training
        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values

        test_x = test[cfg.feature_cols].values
        test_y = test[cfg.label_cols].values

        # Train the model and store the score
        clf.fit(X, y)
        trace.append(clf.score(test_x, test_y))
    print(trace)
    return trace


def ambiguous(train, test, cfg):
    """
    Function to return test scores for ambiguous data appending.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)

    Returns:
    List: Test scores for each iteration
    """

    # Choose the regression method
    clf = get_active_learner(cfg)

    # Initialize the label with a sample from the train data
    label = train.sample(n=cfg.loop.init, random_state=1)  # Fixed random state for reproducibility

    # Remove the sampled label data from the pool
    pool = train.drop(label.index).reset_index(drop=True)

    # Initialize a list to store test scores for each iteration
    number = []

    # Iteratively select data points and evaluate the model
    for i in range(cfg.loop.times):
        # Prepare data for model training
        X_label = label.iloc[:, :-3].values
        y_label = label.iloc[:, -3:].values

        # Fit the model to the current label data
        model = clf.fit(X_label, y_label)

        # Predict probabilities for the pool data
        X_pool = pool.iloc[:, :-3].values
        pred_proba = model.predict_proba(X_pool)[:, 1]

        # Calculate the difference from 0.5 to determine ambiguity
        diff = np.abs(0.333 - pred_proba)

        # Select the top n most ambiguous samples
        top_n_indices = np.argsort(diff)[:cfg.loop.batch_size]

        # Add the selected samples to the label and remove them from the pool
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(pool.index[top_n_indices]).reset_index(drop=True)

        # Evaluate the model on the test data and store the score
        X_test = test.iloc[:, :-3].values
        y_test = test.iloc[:, -3:].values
        score = model.score(X_test, y_test)
        number.append(score)

    return number



def diverse_matrix(train, test, cfg):
    """
    Function to return test scores of diverse append using a distance-based method.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)

    Returns:
    List: Test scores for each iteration
    """

    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    clf = get_active_learner(cfg)
    # Initialize the label with a sample from the train data
    label = train.sample(n=cfg.loop.init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    # Iteratively select data points and evaluate the model
    for i in range(cfg.times):
        # Compute distances between label and pool
        distances = cdist(label, pool)
        minavgdis = []

        # Calculate minimum distance for each point in the pool
        for j in range(distances.shape[1]):
            minavgdis.append(min(distances[:, j]))

        # Convert distances to a NumPy array and sort them
        data_array = np.array(minavgdis)
        sorted_indices = np.argsort(data_array)

        # Select top n indices based on the sorted distances
        top_n_indices = sorted_indices[-cfg.loop.batch_size:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)

        # Prepare data for model training
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]

        # Train the model and store the score
        number.append(clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))

    return number


def diverse_tree(train, test, cfg):
    """
    Function to return test scores of diverse append using a KD-tree based method.

    Parameters:
    train (DataFrame): Training data
    test (DataFrame): Testing data
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)

    Returns:
    List: Test scores for each iteration
    """

    clf = get_active_learner(cfg)
    # Initialize the label with a sample from the train data
    label = train.sample(n=cfg.loop.init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    # Iteratively select data points and evaluate the model
    for _ in range(cfg.loop.times):
        # Create a KD-tree for the current label set
        Tree = cKDTree(label)

        # Query the tree with the pool data to find distances and indices
        distance, _ = Tree.query(pool)

        # Sort the distances
        sort_dis = distance.argsort()

        # Select top n indices based on the sorted distances
        top_n_indices = sort_dis[-cfg.loop.batch_size:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)

        # Prepare data for model training
        X = label.iloc[:, :(label.shape[1] - 3)].values
        y = label.iloc[:, -3:].values

        X_test = test.iloc[:, :(test.shape[1] - 3)].values
        y_test = test.iloc[:, -3:].values

        # Train the model and store the score
        number.append(clf.fit(X, y).score(X_test, y_test))

    return number

if __name__ == '__main__':

    # https://lightning.ai/docs/pytorch/stable/extensions/logging.html#configure-console-logging
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    # ignore all warnings
    simplefilter(action='ignore')

    # loaddata()
    plot()


