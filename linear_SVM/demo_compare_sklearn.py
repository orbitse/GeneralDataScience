"""
This file allows a user to run an experimental comparison between my linear Support Vector Machine
    (SVM) implementation and scikit-learn’s SVM on the Spam data set.
"""
import numpy as np
import pandas as pd
import scipy.linalg
import sklearn.preprocessing
import sklearn.svm

# Get the data
SPAM = pd.read_table(
    'https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.data',
    sep=' ', header=None)
TEST_INDICATOR = pd.read_table(
    'https://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/spam.traintest',
    sep=' ', header=None)

X_DATA = np.asarray(SPAM)[:, 0:-1]
Y_DATA = np.asarray(SPAM)[:, -1]*2 - 1  # Convert to +/- 1
TEST_INDICATOR = np.array(TEST_INDICATOR).T[0]

# Divide the data into train, test sets
x_train = X_DATA[TEST_INDICATOR == 0, :]
x_test = X_DATA[TEST_INDICATOR == 1, :]
y_train = Y_DATA[TEST_INDICATOR == 0]
y_train = y_train.ravel()
y_test = Y_DATA[TEST_INDICATOR == 1]
print("x_train shape is", x_train.shape, "y_train shape is", y_train.shape)

# Standardize Data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Number of samples and the dimension of each sample
n_train = len(y_train)
n_test = len(y_test)
d_feat = np.size(x_data, 1)

lamb = 1
denom = 1/len(y_train)*x_train.T.dot(x_train)
step_init = 1/(scipy.linalg.eigh(denom, eigvals=(d_feat-1, d_feat-1), eigvals_only=True)[0]+lamb)
beta_init = np.zeros(d).ravel()
print("n_train is", n_train, " and  d is", d_feat,
      " and  beta_init.shape is", beta_init.shape)


def computegrad(beta, lamb=1, x=x_train, y=y_train):
    """
       This function computes and returns the gradient for the linear SVM function.
       :param beta: array of model coefficients
       :param lamb: int regularization parameter
       :param x: array of features data
       :param y: array of response data
       """
    n, p = x.shape               # 3450, 57
    grad = np.zeros(x.shape[1])  # initialize gradient vector
    for i in range(0, n):        # row of X_Train & y_train
        right_term = (1 - np.inner((y[i] * x[i, :]), beta))
        grad_row = (-2 * x[i, :].T.dot(y[i]) / n * np.maximum(0, right_term)) + 2 * lamb * beta
        grad = grad + grad_row
    return grad


def objective(beta, lamb=1, x=x_train, y=y_train):
    """
    This function returns the value of the SVM objective function.
    """
    pen = lamb * np.linalg.norm(beta)**2
    return 1/len(y) * np.sum(np.maximum(np.zeros_like(y), (1 - y*np.dot(x, beta)))**2) + pen


def bt_search(beta, lamb=1, step=1, max_iter=50, alpha=0.5, gamma=0.8, x=x_train, y=y_train):
    """
    This function looks for the ideal step size using the backtracking search algorithm.
    Returns optimal step size
    """
    grad_beta = computegrad(beta=beta, lamb=lamb, x=x, y=y)
    norm_grad_beta = np.linalg.norm(grad_beta)
    found_step_size = 0
    iter = 0
    while found_step_size == 0 and iter < max_iter:
        if objective(beta - step * grad_beta, lamb, x, y) <\
                (objective(beta, lamb, x, y) - alpha * step * norm_grad_beta):
            found_step_size = 1
        elif iter == max_iter:
            break
        else:
            step = step*gamma
            #print("Iteration %d. In bt_search, step = %f." % (iter, step))
            iter += 1
    return step


def mylinearsvm(init_step=step_init, max_iter=300, init_beta=beta_init,
                init_theta=np.zeros(d), lamb=1, x=x_train, y=y_train):
    """
    The function implements the fast gradient algorithm to train the linear support
        vector machine with the squared hinge loss
    Inputs: initial step-size value for the backtracking rule and a maximum number of iterations
    """
    beta = init_beta
    theta = init_theta
    beta_vals = beta
    step_size = step_init
    iter = 1
    while iter < max_iter:
        # call backtracking line search function: bt_earch
        step_size = bt_search(beta, lamb, step=step_size, max_iter=50, alpha=0.5,
                              gamma=0.8, x=x, y=y)
        #print("Iteration {:4d}. In mylinearsvm, the step size is{:01.12f}".format(iter, step_size))
        beta = theta - step_size * computegrad(theta, lamb, x=x, y=y)
        # Store all of the places we step to
        beta_vals = np.vstack((beta_vals, beta))
        theta = beta_vals[iter] + iter/(iter+3) * (beta_vals[iter] - beta_vals[iter-1])
        iter += 1
    return beta, beta_vals


def compute_misclassification_error(beta_opt, x, y):
    """
    This function computes and returns the misclassification error based
    on the average percent of misclassification error.
    """
    beta = np.asarray((beta_opt))
    y_pred = (x.dot(beta.T) > 0)*2 - 1
    return round(np.mean(y_pred != y), 3)


opt_betas, all_betas = mylinearsvm(lamb=1)

# Compare to scikit-learn (aka sklearn)
linear_svc = sklearn.svm.LinearSVC(penalty='l2', C=1/(2*lamb*n_train),
                                   fit_intercept=False, tol=10e-8, max_iter=300)
linear_svc.fit(x_train, y_train)

print('Estimated beta values from sklearn:\n', linear_svc.coef_)
print('Estimated beta values with my code:\n', opt_betas)

print("Value of objective function with sklearn's optimal beta values:",
      objective(linear_svc.coef_.flatten()))
print("Value of objective function with optimal beta values from my code:",
      objective(opt_betas))

print("Test set misclassification error with sklearn's optimal beta values and lambda=1:",
      compute_misclassification_error(linear_svc.coef_, x_test, y_test))
print("Test set misclassification error with the optimal beta values from my code and lambda=1:",
      compute_misclassification_error(opt_betas, x_test, y_test))
