# In[8]:


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

import numpy as np
from scipy.special import psi

def loss5(alpha, y):
    """
    customize column numbers
    Compute the gradient of the Dirichlet-Multinomial negative log-likelihood.

    Parameters:
    -----------
    alpha : numpy array of concentration parameters of shape (K,).
    y : numpy array
        Array of observed counts of shape (K,).

    Returns:
    --------
    grad : numpy array
        Gradient vector of shape (K,).
    """
    num_row = y.shape[0]
    num_col = y.shape[1]
    grad, hess = np.zeros_like(y), np.zeros_like(y)

    for i in range(num_row):
        alpha1 = np.array([alpha[num_col*i+j] for j in range(num_col)])
        y1 = y[i]
        psi_sum_alpha = psi(np.sum(y1))
        grad1 = psi_sum_alpha - psi(alpha1 + y1)

        K = len(alpha1)
        psi_prime_alpha = polygamma(1, y1)  # Trigamma function

        hessian1 = np.zeros((K, K))
        for m in range(K):
            for j in range(K):
                if m == j:
                    hessian1[m, j] = psi_prime_alpha[m] - polygamma(1, y1[m] + 1)
                else:
                    hessian1[m, j] = psi_prime_alpha[m]
        grad[i] = grad1
        hess[i] = np.diag(hessian1)

    return grad, hess

def loss6(alpha, y):
    """
    update the two gradient and hessian functions
    customize column numbers
    Compute the gradient of the Dirichlet-Multinomial negative log-likelihood.

    Parameters:
    -----------
    alpha : numpy array of concentration parameters of shape (K,).
    y : numpy array
        Array of observed counts of shape (K,).

    Returns:
    --------
    grad : numpy array
        Gradient vector of shape (K,).
    """
    epsilon = 1
    num_row = y.shape[0]
    num_col = y.shape[1]
    grad, hess = np.zeros_like(y), np.zeros_like(y)

    for i in range(num_row):
        alpha1 = np.array([alpha[num_col*i+j] for j in range(num_col)])
        y1 = y[i]
        K = len(alpha1)
        N = np.sum(y1)
        alpha_sum = np.sum(alpha1)
        n_alpha_sum = N + alpha_sum

        # Gradient computation
        gradient = np.zeros(K)
        for m in range(K):
            gradient[m] = -(
                    psi(n_alpha_sum) - psi(alpha_sum)
                    - psi(y1[m] + alpha1[m]) + psi(alpha1[m])
            )

        # Hessian computation
        hessian = np.zeros((K, K))
        psi_prime_n_alpha_sum = polygamma(1, n_alpha_sum)
        psi_prime_alpha_sum = polygamma(1, alpha_sum)

        for s in range(K):
            for j in range(K):
                if s == j:
                    hessian[s, j] = -(
                            psi_prime_n_alpha_sum - psi_prime_alpha_sum
                            - polygamma(1, y1[s] + alpha1[s]) + polygamma(1, alpha1[s])
                    )
                else:
                    hessian[s, j] = -psi_prime_n_alpha_sum + psi_prime_alpha_sum
        grad[i] = gradient
        hess[i] = np.diag(hessian)

    return grad, hess

def loss7(alpha, y):
    """
    trying to fix infinity
    update the two gradient and hessian functions
    customize column numbers
    Compute the gradient of the Dirichlet-Multinomial negative log-likelihood.

    Parameters:
    -----------
    alpha : numpy array of concentration parameters of shape (K,).
    y : numpy array
        Array of observed counts of shape (K,).

    Returns:
    --------
    grad : numpy array
        Gradient vector of shape (K,).
    """
    epsilon = 1
    num_row = y.shape[0]
    num_col = y.shape[1]
    grad, hess = np.zeros_like(y), np.zeros_like(y)

    for i in range(num_row):
        alpha1 = np.array([alpha[num_col*i+j] for j in range(num_col)])
        y1 = y[i]
        alpha1 = np.maximum(alpha1, epsilon)
        y1 = np.maximum(y1, epsilon)
        K = len(alpha1)
        N = np.sum(y1)
        alpha_sum = np.sum(alpha1)
        n_alpha_sum = N + alpha_sum

        # Gradient computation
        gradient = np.zeros(K)
        for m in range(K):
            gradient[m] = -(
                    psi(n_alpha_sum) - psi(alpha_sum)
                    - psi(y1[m] + alpha1[m]) + psi(alpha1[m])
            )

        # Hessian computation
        hessian = np.zeros((K, K))
        psi_prime_n_alpha_sum = polygamma(1, n_alpha_sum)
        psi_prime_alpha_sum = polygamma(1, alpha_sum)

        for s in range(K):
            for j in range(K):
                if s == j:
                    hessian[s, j] = -(
                            psi_prime_n_alpha_sum - psi_prime_alpha_sum
                            - polygamma(1, y1[s] + alpha1[s]) + polygamma(1, alpha1[s])
                    )
                else:
                    hessian[s, j] = -psi_prime_n_alpha_sum + psi_prime_alpha_sum
        grad[i] = gradient
        hess[i] = np.diag(hessian)

    return grad, hess

def loss8(y, alpha):
    """
    trying to fix infinity, switch the order of y and alpha (what does they represents really?)
    update the two gradient and hessian functions
    customize column numbers
    Compute the gradient of the Dirichlet-Multinomial negative log-likelihood.

    Parameters:
    -----------
    alpha : numpy array of concentration parameters of shape (K,).
    y : numpy array
        Array of observed counts of shape (K,).

    Returns:
    --------
    grad : numpy array
        Gradient vector of shape (K,).
    """
    epsilon = 1
    num_row = alpha.shape[0]
    num_col = alpha.shape[1]
    grad, hess = np.zeros_like(alpha), np.zeros_like(alpha)

    for i in range(num_row):
        y1 = np.array([y[num_col*i+j] for j in range(num_col)])
        alpha1 = alpha[i]
        alpha1 = np.maximum(alpha1, epsilon)
        y1 = np.maximum(y1, epsilon)
        K = len(y1)
        N = np.sum(alpha1)
        y_sum = np.sum(y1)
        n_alpha_sum = N + y_sum

        # Gradient computation
        gradient = np.zeros(K)
        for m in range(K):
            gradient[m] = -(
                    psi(n_alpha_sum) - psi(y_sum)
                    - psi(y1[m] + alpha1[m]) + psi(y1[m])
            )

        # Hessian computation
        hessian = np.zeros((K, K))
        psi_prime_n_alpha_sum = polygamma(1, n_alpha_sum)
        psi_prime_alpha_sum = polygamma(1, y_sum)

        for s in range(K):
            for j in range(K):
                if s == j:
                    hessian[s, j] = -(
                            psi_prime_n_alpha_sum - psi_prime_alpha_sum
                            - polygamma(1, y1[s] + alpha1[s]) + polygamma(1, y1[s])
                    )
                else:
                    hessian[s, j] = -psi_prime_n_alpha_sum + psi_prime_alpha_sum
        grad[i] = gradient
        hess[i] = np.diag(hessian)

    return grad, hess


'''
>>> x = torch.autograd.functional.hessian(func= get_dirichlet_neg_log_prob,inputs= (w, torch.tensor(6.), b))
>>> x[2][2]
'''
'''
>>> loss = get_dirichlet_neg_log_prob(w, torch.tensor(6), b)
>>> loss.backward()
>>> print(b.grad)
tensor([ 0.0337, -0.0496, -0.0830])
'''

def loss10(y, alpha):
    """
    """
    epsilon = 1
    num_row = alpha.shape[0]
    num_col = alpha.shape[1]
    y1 = np.zeros_like(alpha)
    for i in range(num_row):
        y1[i] = np.array([y[num_col * i + j] for j in range(num_col)])
    grad, hess = np.zeros_like(alpha), np.zeros_like(alpha)
    for i in range(num_row):
        actual = torch.tensor(y1[i], requires_grad=True)
        pred = torch.tensor(alpha[i], requires_grad=True)
        loss_fun = get_dirichlet_neg_log_prob(actual, actual.sum(), pred)
        loss_fun.backward()
        grad[i] = np.array(pred.grad)
        total_hess = torch.autograd.functional.hessian(func=get_dirichlet_neg_log_prob, inputs=(actual, actual.sum(), pred))
        hess[i] = np.diag(total_hess[num_col-1][num_col-1])
    return grad, hess

def loss11(y, alpha):
    """
    """
    epsilon = 1
    num_row = alpha.shape[0]
    num_col = alpha.shape[1]
    y1 = np.zeros_like(alpha)
    for i in range(num_row):
        y1[i] = np.array([y[num_col * i + j] for j in range(num_col)])
    grad, hess = np.zeros_like(alpha), np.zeros_like(alpha)
    for i in range(num_row):
        a = torch.tensor(y1[i])
        b = torch.tensor(alpha[i])
        actual = torch.nn.functional.softplus(a)
        pred = torch.nn.functional.softplus(b)
        actual.requires_grad_(True)
        pred.requires_grad_(True)
        loss_fun = get_dirichlet_neg_log_prob(actual, actual.sum(), pred)
        loss_fun.backward()
        grad[i] = np.array(pred.grad)
        total_hess = torch.autograd.functional.hessian(func=get_dirichlet_neg_log_prob, inputs=(actual, actual.sum(), pred))
        hess[i] = np.diag(total_hess[num_col-1][num_col-1])
    return grad, hess

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    parameters = {"n_estimators": 100,
                  'objective': loss11,
                  'learning_rate': 0.02,
                  'max_depth': 5,
                  'base_score': 10}
    model = XGBRegressor(**parameters)
    x = loaddata().iloc[:, :-4]
    y = loaddata().iloc[:, -4:-1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)
    model = model.fit(X_train, y_train)
    print(model.predict(X_test))

