# THIS IS A FINAL FILE FOR THE COMPUTATIONS WITHIN 4.4



import math
import numpy as np
import scipy
import datetime

from vkoga import kernels
from Allen_Cahn.Tizian.utils import ValueFunction
from scipy.spatial import distance_matrix
import time
from matplotlib import pyplot as plt
from vkoga_pde.kernels_PDE import Gaussian_laplace

import scipy.io


desired_width=400 # 320
np.set_printoptions(linewidth=desired_width)


flag_with_gradient = True


## Load dataset
array_X = np.load('X_FINAL.npy')
array_values = np.load('array_values_FINAL.npy').reshape(-1, 1)

X_train = array_X
y_train = array_values


## Settings
# General settings
dim = 30

# Kernel settings # TODO: Update me!
shape_para = 1/np.sqrt(dim)
reg_para = 1e-10

rbf = lambda r: np.exp(-r**2)
rbf_dash = lambda r: -2 * r * np.exp(-r**2)                     # derivative of Gaussian kernel
rbf_dash_divided_by_r = lambda r: -2 * np.exp(-r**2)            # derivative of Gaussian kernel

## Compute kernel model
if flag_with_gradient:
    kernel = Gaussian_laplace(dim=dim, ep=shape_para)


    ctrs = np.concatenate((np.zeros((dim, dim)), X_train.copy()), axis=0)   # add zeros to enforce gradients = 0 in origin
    f_train = np.concatenate((np.zeros((dim, 1)), y_train.copy()), axis=0)     # add zeros to enforce gradients = 0 in origin

    array_n = np.eye(dim)           # array for the normal vectors

    # build (generalized) kernel matrix
    A = np.zeros((ctrs.shape[0], ctrs.shape[0]))

    A[:dim, :dim] = kernel.mixed_n_n(ctrs[:dim, :], ctrs[:dim, :], array_n, array_n)
    A[:dim, dim:] = kernel.mixed_k_n(ctrs[dim:, :], ctrs[:dim, :], array_n).T
    A[dim:, :dim] = kernel.mixed_k_n(ctrs[dim:, :], ctrs[:dim, :], array_n)
    A[dim:, dim:] = kernel.eval(ctrs[dim:, :], ctrs[dim:, :])

    A0 = A + reg_para * np.eye(A.shape[0])

    coeff = np.linalg.solve(A0, f_train)

    model_numpy1 = lambda x: kernel.mixed_k_n(x, ctrs[:dim, :], array_n) @ coeff[:dim] 
    model_numpy2 = lambda x: kernel.eval(x, ctrs[dim:, :]) @ coeff[dim:]

    model_numpy = lambda x: model_numpy1(x) + model_numpy2(x)
else:
    kernel = kernels.Gaussian(ep=shape_para)

    ctrs = X_train.copy()
    
    t0_train = time.time()
    A0 = kernel.eval(ctrs, ctrs) + reg_para * np.eye(ctrs.shape[0])
    coeff = np.linalg.solve(A0, y_train)
    t1_train = time.time()

    model_numpy = lambda x: kernel.eval(x, ctrs) @ coeff

# Compute training loss
loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)

train_loss = loss_rel(y_train, model_numpy(X_train))
print('Train loss: {:.3e}'.format(train_loss))


if flag_with_gradient:

    # Gradient due to the Riesz represnter for the dim many derivatives (gradient = 0 in the origin)
    grad_kernel_model1 = lambda x: rbf(-shape_para * np.linalg.norm(x)) * \
        (2 * shape_para**2 * coeff[:dim] - 4 * shape_para**4 * np.atleast_2d(x).T * (np.atleast_2d(x) @ coeff[:dim]))

    # Gradient due to the Riesz representer of the interpolatino points
    grad_kernel_model2 = lambda x: shape_para**2 * ((np.atleast_2d(x).T - np.atleast_2d(ctrs[dim:]).T) * 
        rbf_dash_divided_by_r(shape_para * distance_matrix(np.atleast_2d(x), np.atleast_2d(ctrs[dim:])))) @ coeff[dim:]

    grad_kernel_model = lambda x: grad_kernel_model1(x) + grad_kernel_model2(x)
    
else:
    # Assume only one data point!!! # check gradients via torch and finite differences!
    grad_kernel_model = lambda x: shape_para**2 * ((np.atleast_2d(x).T - np.atleast_2d(ctrs).T) * 
        rbf_dash_divided_by_r(shape_para * distance_matrix(np.atleast_2d(x), np.atleast_2d(ctrs)))) @ coeff






## Optimal control stuff
sigma = 1.e-2               # some parameter
N = 30                      # dimension
t0 = 0                      # initial time
t1 = 60                    # final time
dt = .01                  # temporal discretization step           # was 0.001 before

x = np.linspace(0,1,N)
dx = x[2]-x[1]              # spatial discretization step

I = np.eye(N)

# Build matrix A0 and A
A0 = -2*np.eye(N)+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1)
A0[0,0] = -1; # Neumann
A0[-1,-1] = -1

A = sigma*A0/dx**2          # A is the discretized Laplace operator
gamma = dx

Ax = lambda x: A+np.diag(1-x**2)
P_sdre = lambda x: gamma*(scipy.linalg.sqrtm(dx * I / gamma + Ax(x) @ Ax(x)) + Ax(x))
v = lambda x: x.T @ P_sdre(x) @ x


## Compute test inputs
value_beta = 3.0
array_a = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
X_test = np.zeros((array_a.shape[0], len(x)))

for idx_a in range(array_a.shape[0]):

    array_k = np.arange(1, 5).reshape(1, -1)
    X_test[idx_a, :] = (array_a[[idx_a], :]) @ (array_k.T**(-value_beta) * np.cos(2 * math.pi * array_k.T * x.reshape(1, -1)))

# normalize
X_test = X_test / 2




## Compute cost on test points
for idx in [0, 1, 2, 3]: #range(4):
    
    print(' ')
    print(' ')
    print('Comparison of value function and surrogate approx on test points')

    v_true = v(X_test[idx, :].T)    
    v_surr = model_numpy(X_test[[idx], :]).item()

    print('{:.4f}, {:.4f}, {:.4e}'.format(v_true, v_surr, v_true - v_surr))

    print(' ')

    # X[0, :] = X_train[idx, :]

    # Compute value function
    valFunc_sdre = ValueFunction(t0, t1, dt, dx, gamma, lambda x: P_sdre(x), lambda x: Ax(x), str_control='sdre')
    total_cost_sdre, list_y_sdre = valFunc_sdre.value_function_mathias(X_test[idx,:])

    # Compute value function using the kernel surrogate
    valFunc_surr = ValueFunction(t0, t1, dt, dx, gamma, lambda x: P_sdre(x), lambda x: Ax(x), 
                                lambda x: grad_kernel_model(x), str_control='surr')
    total_cost_surr, list_y_surr = valFunc_surr.value_function_mathias(X_test[idx,:])

    print('cost sdre = {:.4f}, cost surr = {:.4f}, diff = {:.4f}'.format(
        total_cost_sdre, total_cost_surr, total_cost_sdre - total_cost_surr))
    # print(total_cost_sdre, total_cost_surr, total_cost_sdre - total_cost_surr)

    



## Some visualization
plt.figure(10000)
plt.clf()
for idx in np.linspace(0, 300, 10):
    idx = int(idx)
    plt.plot(list_y_sdre[idx], 'x-', label='time={:.3f}'.format(idx*dt))
plt.legend()
plt.title('SDRE')
plt.show(block=False)

plt.figure(10001)
plt.clf()
for idx in np.linspace(0, 300, 10):
    idx = int(idx)
    plt.plot(list_y_surr[idx], 'x-', label='time={:.3f}'.format(idx*dt))
plt.legend()
plt.title('SURR')
plt.show(block=False)




array_sdre = np.stack(list_y_sdre)
array_surr = np.stack(list_y_surr)




