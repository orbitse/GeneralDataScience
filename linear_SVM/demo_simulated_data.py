import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn.preprocessing
import sklearn.svm

# Create the simulated data
np.random.seed(10)
# create 2D array with 120 rows & 50 columns & initially each element equal to zero
data = np.zeros((120, 50))
# each class has 60 observations
data[0:40, :] = np.random.normal(loc=25, scale=5, size=(40, 50))
data[40:80, :] = np.random.normal(loc=50, scale=5, size=(40, 50))
data[80:120, :] = np.random.normal(loc=75, scale=5, size=(40, 50))
# these are the class label values: 60 are -1 and 60 are 1
classes = [-1]*60 + [1]*60
# centers the data so expected value = 0 and no intercept/bias is needed
data_centered = data - np.mean(data, axis=0)

# Divide the data into train, test sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, classes, random_state=0)
y_train = np.asarray(y_train)
print("x_train shape is", x_train.shape, "y_train shape is", y_train.shape)

# Standardize Data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Number of samples and the dimension of each sample
n_train = len(y_train)
n_test = len(y_test)
d = np.size(x_train, 1)
beta_init = np.zeros(d).ravel()
print("n_train is", n_train, " and  d is", d, " and  beta_init.shape is", beta_init.shape)


def computegrad(beta, lamb, x=x_train, y=y_train):
    """
    Computes and returns the gradient for the linear SVM function.
    :param beta: array of model coefficients
    :param lamb: int regularization parameter
    :param x: array of features data
    :param y: array of response data
    """
    penalty = 2*lamb*beta
    gradient = -2/np.size(x, 0)*np.sum(y[:, np.newaxis]*x*np.max((np.zeros_like(y),
                                                              1-y*np.dot(x, beta)),
                                                             axis=0)[:, np.newaxis], axis=0) + penalty
    return gradient


def objective(beta, lamb=1, x=x_train, y=y_train):
    """
    This function returns the value of the linear SVM objective function.
    """
    pen = lamb * np.linalg.norm(beta)**2
    return 1/len(y) * np.sum(np.maximum(y, 1 - y*np.dot(x, beta))**2) + pen


def bt_search(beta, lamb=1, step=1, max_iter=50, alpha=0.5, gamma=0.8, x=x_train, y=y_train):
    """
    This function looks for the ideal step size using the backtracking search algorithm.
    Returns optimal step size
    """
    grad_beta = computegrad(beta, lamb, x, y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_step_size = 0
    iter = 0
    while(found_step_size == 0 and iter < max_iter):
        if objective(beta - step * grad_beta, lamb, x, y)\
                < (objective(beta, lamb, x, y) - alpha * step * norm_grad_beta):
            found_step_size = 1
        elif iter == max_iter:
            break
        else:
            step = step*gamma
            iter += 1
    return step


def mylinearsvm(init_step_par=None, max_iter=150, init_beta=beta_init,
                init_theta=np.zeros(d), lamb=1, x=x_train, y=y_train):
    """
    The function implements the fast gradient algorithm to train the linear support
        vector machine with the squared hinge loss
    Inputs: initial step-size value for the backtracking rule and a maximum number of iterations
    """
    beta = init_beta
    theta = init_theta
    beta_vals = beta
    if init_step_par is None:
        denom = 1 / len(y_train) * x_train.T.dot(x_train)
        step_size = 1 / (scipy.linalg.eigh(denom, eigvals=(d - 1, d - 1), eigvals_only=True)[0] + lamb)
    else:
        step_size = init_step_par
    iter = 1
    while iter < max_iter:
        step_size = bt_search(beta, lamb, step=step_size, max_iter=30, alpha=0.5,
                              gamma=0.8, x=x, y=y)
        beta = theta - step_size * computegrad(theta, lamb, x, y)
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta))
        theta = beta_vals[iter] + iter/(iter+3) * (beta_vals[iter] - beta_vals[iter-1])
        iter += 1
    return beta, beta_vals

opt_betas, all_betas = mylinearsvm(lamb=1)
print("\nThe betas from mylinearsvm with Lambda = 1 are\n", opt_betas)


def objective_plot(betas, lamb=1, x=x_train, y=y_train, save_file=''):
    """
    Plot the values of the objective function.
    :param betas: coefficients
    :param lamb: regularization parameter
    :param x: features data
    :param y: labels data
    :param save_file: 
    """
    num_points = np.size(betas, 0)
    objs = np.zeros(num_points)
    for i in range(0, num_points):
        objs[i] = objective(betas[i, :], lamb, x=x, y=y)
    fig = plt.figure(figsize=(4, 4), dpi=150)
    plt.plot(range(1, num_points + 1), objs, color="cornflowerblue", linewidth=1.8,
             linestyle="dashed", marker="o", label='SVM Fast Gradient')
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function Value")
    plt.title("Objective Values with Lambda = "+str(lamb))
    plt.legend(loc="upper right")
    if not save_file:
        plt.show()
    else:
        plt.savefig(save_file)

objective_plot(all_betas)

