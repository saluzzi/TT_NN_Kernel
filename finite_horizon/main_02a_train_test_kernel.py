# Based on the data created in main_01a_create_train_test_set.py, we will train a kernel 
# model to approximate the data set.


import math
import numpy as np
import scipy
import datetime

from vkoga_2L import kernels
from Allen_Cahn.Tizian.utils import ValueFunction
from scipy.spatial import distance_matrix
import time
from matplotlib import pyplot as plt
from vkoga_PDE.kernels_PDE import Gaussian_laplace


desired_width=400 # 320
np.set_printoptions(linewidth=desired_width)


flag_with_gradient = False      # True automatically uses the Gaussian kernel, for polynomial not implemented


## Load dataset
X_train = np.load('X_train_finite_horizon.npy')
X_test = np.load('X_test_finite_horizon.npy')
y_train = np.load('array_values_train_finite_horizon.npy')
y_test = np.load('array_values_test_finite_horizon.npy')


## Settings
# General settings
dim = 30

para_a = .1
para_p = 2

# Kernel settings
shape_para = .05/np.sqrt(dim)
reg_para = 1e-10


## Compute kernel model
if flag_with_gradient:
    kernel = Gaussian_laplace(dim=dim, ep=shape_para)


    ctrs = np.concatenate((np.zeros((dim, dim)), X_train.copy()), axis=0)   # add zeros to enforce gradients = 0 in origin
    f_train = np.concatenate((np.zeros((dim, 1)), y_train.copy()), axis=0)     # add zeros to enforce gradients = 0 in origin

    array_n = np.eye(dim)           # array for the normal vectors

    # build (generalized) kernel matrix
    t0_train = time.time()
    A = np.zeros((ctrs.shape[0], ctrs.shape[0]))

    A[:dim, :dim] = kernel.mixed_n_n(ctrs[:dim, :], ctrs[:dim, :], array_n, array_n)
    A[:dim, dim:] = kernel.mixed_k_n(ctrs[dim:, :], ctrs[:dim, :], array_n).T
    A[dim:, :dim] = kernel.mixed_k_n(ctrs[dim:, :], ctrs[:dim, :], array_n)
    A[dim:, dim:] = kernel.eval(ctrs[dim:, :], ctrs[dim:, :])

    A0 = A + reg_para * np.eye(A.shape[0])

    coeff = np.linalg.solve(A0, f_train)
    t1_train = time.time()

    model_numpy1 = lambda x: kernel.mixed_k_n(x, ctrs[:dim, :], array_n) @ coeff[:dim] 
    model_numpy2 = lambda x: kernel.eval(x, ctrs[dim:, :]) @ coeff[dim:]

    model_numpy = lambda x: model_numpy1(x) + model_numpy2(x)
else:
    kernel = kernels.Gaussian(ep=shape_para)
    # kernel = kernels.Polynomial(a=para_a, p=para_p)

    ctrs = X_train.copy()
    
    t0_train = time.time()
    A0 = kernel.eval(ctrs, ctrs) + reg_para * np.eye(ctrs.shape[0])
    coeff = np.linalg.solve(A0, y_train)
    t1_train = time.time()

    model_numpy = lambda x: kernel.eval(x, ctrs) @ coeff

# Compute training loss
loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)

t0 = time.time()
y_train_pred = model_numpy(X_train)
t1 = time.time()
y_test_pred = model_numpy(X_test)
t2 = time.time()


train_loss = loss_rel(y_train, y_train_pred)
test_loss = loss_rel(y_test, y_test_pred)




print('Train loss: {:.3e}. Training prediction for {} points took {:.3e}s. Training took {:.3e}s.'.format(
    train_loss, X_train.shape[0], t1-t0, t1_train-t0_train))
print('Test loss:  {:.3e}. Test prediction for {} points took {:.3e}s.'.format(test_loss, X_test.shape[0], t2-t1))














