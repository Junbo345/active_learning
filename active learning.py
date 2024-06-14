# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# ignore all warnings
simplefilter(action='ignore')




def loaddata():
    df = pd.read_csv(r'C:\desi_representations_subset.csv') # change the path based on different computers
    # load the data
    columns_to_select = ['feat_pca_{}'.format(i) for i in range(20)] + ['classification_label']
    X = df[columns_to_select]
    return X




def plot(times=100, init=50, loop=1, regression_method="L"):
    """
    Function to plot the performance of different sampling methods over iterations.

    Parameters:
    times (int): Number of iterations (default = 100)
    init (int): Initial pool of data (default = 50)
    loop (int): Number of data points in each batch of iteration (default = 1)
    regression_method (str): Method for regression ('L' for Logistic Regression, 'M' for MLPClassifier)
    """

    # Load and split the data into test and train sets
    test = loaddata().sample(n=2000, random_state=1)
    train = loaddata().drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)

    # Initialize the X-axis values for the plot
    x = []
    for i in range(times):
        x.append(i)
    x = np.array(x)

    # Choose the regression method
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    # Prepare training data for initial fitting
    X = train.iloc[:, :(train.shape[1] - 1)]
    y = train.iloc[:, -1]

    # Calculate the Y-axis values for different sampling methods
    y1 = abs(np.log(1 - np.array(randomapp(train, test, times=times, loop=loop, init=init, regression_method=regression_method))))
    y2 = abs(np.log(1 - np.array(ambiguous(train, test, times=times, loop=loop, init=init, regression_method=regression_method))))
    y3 = abs(np.log(1 - np.array(diverse_tree(train, test, times=times, loop=loop, init=init, regression_method=regression_method))))

    # Plot a horizontal line representing the baseline score
    plt.axhline(y=abs(math.log(1 - clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))), color='r', linestyle='-')

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
    title = (f'Three Lines with Different Methods\nIterations: {times}, Initial sample size: {init}, '
             f'Batch size: {loop}, Regression Method: {regression_method}')
    plt.title(title)

    # Display the legend
    plt.legend()

    # Show the plot
    return plt.show()


def randomapp(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
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

    # Check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    # Check the type of times
    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    # Check the type of loop
    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    # Choose the regression method
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    # Initialize the label with a sample from the train data
    label = train.sample(n=init)

    # Remove the sampled label data from the pool
    pool = train.drop(label.index)

    # Initialize a list to store test scores for each iteration
    trace = []

    # Iteratively select data points and evaluate the model
    for i in range(times):
        # Randomly sample data points from the pool
        new = pool.sample(n=loop)

        # Add the new samples to the label
        label = pd.concat([label, new])

        # Remove the new samples from the pool
        pool = pool.drop(new.index)

        # Prepare data for model training
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]

        # Train the model and store the score
        trace.append(clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))

    return trace


def ambiguous(train, test, times=100, init=50, loop=1, regression_method="L"):
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

    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    # Check the type of times
    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    # Check the type of loop
    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    # Choose the regression method
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    # Initialize the label with a sample from the train data
    label = train.sample(n=init, random_state=1)  # Fixed random state for reproducibility

    # Remove the sampled label data from the pool
    pool = train.drop(label.index).reset_index(drop=True)

    # Initialize a list to store test scores for each iteration
    number = []

    # Iteratively select data points and evaluate the model
    for i in range(times):
        # Prepare data for model training
        X_label = label.iloc[:, :-1]
        y_label = label.iloc[:, -1]

        # Fit the model to the current label data
        model = clf.fit(X_label, y_label)

        # Predict probabilities for the pool data
        X_pool = pool.iloc[:, :-1]
        pred_proba = model.predict_proba(X_pool)[:, 1]

        # Calculate the difference from 0.5 to determine ambiguity
        diff = np.abs(0.5 - pred_proba)

        # Select the top n most ambiguous samples
        top_n_indices = np.argsort(diff)[:loop]

        # Add the selected samples to the label and remove them from the pool
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(pool.index[top_n_indices]).reset_index(drop=True)

        # Evaluate the model on the test data and store the score
        X_test = test.iloc[:, :-1]
        y_test = test.iloc[:, -1]
        score = model.score(X_test, y_test)
        number.append(score)

    return number



def diverse_matrix(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
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

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    # Choose the regression method
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    # Initialize the label with a sample from the train data
    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    # Iteratively select data points and evaluate the model
    for i in range(times):
        # Compute distances between label and pool
        distances = cdist(label, pool)
        minavgdis = []

        # Calculate minimum distance for each point in the pool
        for j in range(distances.shape[1]):
            minavgdis.append(min(distances[:, j]))

        # Convert distances to a NumPy array and sort them
        data_array = np.array(minavgdis)
        sorted_indices = np.argsort(data_array)
        n = loop

        # Select top n indices based on the sorted distances
        top_n_indices = sorted_indices[-n:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)

        # Prepare data for model training
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]

        # Train the model and store the score
        number.append(clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))

    return number


def diverse_tree(train=None, test=None, times=100, init=50, loop=1, regression_method="L"):
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

    # Check if train and test data are provided
    if train is None or test is None:
        raise ValueError("Both train and test data must be provided.")

    # Debug statement to check the type of init
    if not isinstance(init, int):
        raise TypeError(f"Expected init to be an integer, but got {type(init)} instead.")

    if not isinstance(times, int):
        raise TypeError(f"Expected times to be an integer, but got {type(times)} instead.")

    if not isinstance(loop, int):
        raise TypeError(f"Expected loop to be an integer, but got {type(loop)} instead.")

    # Choose the regression method
    if regression_method == "L":
        clf = LogisticRegression()
    elif regression_method == "M":
        clf = MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.01)

    # Initialize the label with a sample from the train data
    label = train.sample(n=init)
    pool = train.drop(label.index).reset_index(drop=True)
    number = []

    # Iteratively select data points and evaluate the model
    for i in range(times):
        # Create a KD-tree for the current label set
        Tree = cKDTree(label)

        # Query the tree with the pool data to find distances and indices
        distance, ind = Tree.query(pool)

        # Sort the distances
        sort_dis = distance.argsort()

        # Select top n indices based on the sorted distances
        top_n_indices = sort_dis[-loop:]
        label = pd.concat([label, pool.iloc[top_n_indices]], ignore_index=True)
        pool = pool.drop(top_n_indices).reset_index(drop=True)

        # Prepare data for model training
        X = label.iloc[:, :(label.shape[1] - 1)]
        y = label.iloc[:, -1]

        # Train the model and store the score
        number.append(clf.fit(X, y).score(test.iloc[:, :(test.shape[1] - 1)], test.iloc[:, -1]))

    return number
