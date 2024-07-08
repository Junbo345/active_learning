import numpy as np
import pandas as pd
import pyro
import matplotlib.pyplot as plt
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from scipy.special import gammaln
from sympy.physics.continuum_mechanics.beam import numpy
from xgboost import XGBRegressor
from scipy.special import psi, polygamma

# ignore all warnings
simplefilter(action='ignore')





def loaddata():
    df = pd.read_csv(r'C:\Users\20199\OneDrive\Desktop\desi_representations_subset.csv') # change the path based on different computers
    # load the data
    columns_to_select = ['feat_pca_{}'.format(i) for i in range(20)] + ['smooth-or-featured_featured-or-disk'] + ['smooth-or-featured_smooth'] + ['smooth-or-featured_artifact'] + ['smooth-or-featured_total-votes']
    return df[columns_to_select]
 # In[0]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.gamma import Gamma
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.multinomial import Multinomial

def get_dirichlet_neg_log_prob(labels_for_q, total_count, concentrations_for_q):
    # https://docs.pyro.ai/en/stable/distributions.html#dirichletmultinomial
    # .int()s avoid rounding errors causing loss of around 1e-5 for questions with 0 votes
    dist = pyro.distributions.DirichletMultinomial(
        total_count=total_count.int(), concentration=concentrations_for_q, is_sparse=False, validate_args=True)
    return -dist.log_prob(labels_for_q.int())  # important minus sign


def loss(y, alpha):
    """
    This function calculates the gradient and Hessian of a custom loss function
    for Dirichlet distribution parameters using PyTorch's automatic differentiation.

    Parameters:
    y (numpy array): Flattened target array.
    alpha (numpy array): Array of predicted Dirichlet distribution parameters.

    Returns:
    grad (numpy array): Gradient of the loss function with respect to alpha.
    hess (numpy array): Hessian of the loss function with respect to alpha.
    """
    num_row = alpha.shape[0]  # Number of rows in the alpha array
    num_col = alpha.shape[1]  # Number of columns in the alpha array

    # Initialize y1 with the same shape as alpha and fill it with zeros
    y1 = np.zeros_like(alpha)

    # Reshape the flattened target array y into the same shape as alpha
    for i in range(num_row):
        y1[i] = np.array([y[num_col * i + j] for j in range(num_col)])

    # Initialize gradient and Hessian arrays with the same shape as alpha
    grad, hess = np.zeros_like(alpha), np.zeros_like(alpha)

    # Iterate through each row of the reshaped target array y1 and alpha
    for i in range(num_row):
        a = torch.tensor(y1[i])  # Convert the ith row of y1 to a PyTorch tensor
        b = torch.tensor(alpha[i])  # Convert the ith row of alpha to a PyTorch tensor

        # Apply the softplus function to the tensors
        actual = torch.nn.functional.softplus(a)
        pred = torch.nn.functional.softplus(b)

        # Enable gradient computation for actual and pred tensors
        actual.requires_grad_(True)
        pred.requires_grad_(True)

        # Calculate the negative log probability loss for Dirichlet distribution
        loss_fun = get_dirichlet_neg_log_prob(actual, actual.sum(), pred)

        # Perform backpropagation to compute the gradients
        loss_fun.backward()

        # Store the gradients of pred in the grad array
        grad[i] = np.array(pred.grad)

        # Compute the Hessian of the loss function with respect to actual and pred
        total_hess = torch.autograd.functional.hessian(func=get_dirichlet_neg_log_prob,
                                                       inputs=(actual, actual.sum(), pred))

        # Store the diagonal elements of the Hessian in the hess array
        hess[i] = np.diag(total_hess[num_col - 1][num_col - 1])

    # Return the computed gradients and Hessians
    return grad, hess

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    parameters = {"n_estimators": 100,
                  'objective': loss,
                  'learning_rate': 0.02,
                  'max_depth': 5,
                  'base_score': 10}
    model = XGBRegressor(**parameters)
    x = loaddata().iloc[:, :-4]
    y = loaddata().iloc[:, -4:-1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)
    model = model.fit(X_train, y_train)
    print(model.predict(X_test))
