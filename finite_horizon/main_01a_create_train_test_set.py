# Based on the google colab provided by Luca, we compute training and test data
# to compute and evaluate our ML models


import numpy as np
from scipy.sparse import eye as speye, diags, csr_matrix
from scipy.integrate import solve_ivp
import time
from datetime import datetime

# Initialization
N = 30
sigma = 1e-2
x = np.linspace(0, 1, N)
dx = x[1] - x[0]
T = 1

# Constructing A0 matrix
A0 = -2 * np.eye(N) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
A0[0, 1] = 2  # Neumann boundary condition
A0[-1, -2] = 2

# Sparse matrix A
A = csr_matrix(sigma * A0 / dx**2)

# Identity matrix B
B = speye(N)

# Matrices for control problem
Ptf = dx * speye(N)
Rxx = Ptf
Ruu = Ptf
BB = B @ np.linalg.inv(Ruu.toarray()) @ B.toarray()

# Function definitions
def doty(t, y, A, BB, Rx):
    P = unvec(y)
    dotP = P @ A + A.T @ P + Rx.toarray() - P @ BB @ P
    return vec(dotP)

def vec(P):
    y = []
    for i in range(len(P)):
        y.append(P[i, i:])
    return np.concatenate(y)

def unvec(y):
    N = int(np.roots([1, 1, -2 * len(y)]).max())
    P = np.zeros((N, N))
    kk = N
    kk0 = 0
    for ii in range(N):
        P[ii, ii:N] = y[kk0:kk0+kk]
        kk0 += kk
        kk -= 1
    return (P + P.T) - np.diag(np.diag(P))

def riccati(t, dot_P, Ptf, A, BB, Rx, T, options=None):
    # Wrap the dot_P function to include additional arguments
    def wrapped_dot_P(t, y):
        return dot_P(t, y, A, BB, Rx)
    if abs(t-T)<=1.e-6:
        sol = vec(Ptf.toarray())
        return unvec(sol)
    else:
      sol = solve_ivp(wrapped_dot_P, [t, T], vec(Ptf.toarray()), method='RK45', t_eval=[t])
      return unvec(sol.y[:, 0])

# Function that uses riccati
def fun(x, A, BB, Rx, T):
    P0 = riccati(x[0], doty, Ptf, A, BB, Rx, T)
    return np.dot(x[1:], np.dot(P0, x[1:]))



# Create training data and test data
n_train = 5000
n_test = 5000
X_train = np.random.rand(n_train, N + 1)
X_test = np.random.rand(n_test, N + 1)


# Compute training and test values values
list_values_train = []
for idx_pt in range(X_train.shape[0]):
    if idx_pt % 100 == 0:
        print(datetime.now().strftime("%H:%M:%S"), 'Computation of training data {}/{}.'.format(idx_pt+1, n_train))
    value = fun(X_train[idx_pt, :], A, BB, Rxx, T)
    list_values_train.append(value)

list_values_test = []
for idx_pt in range(X_test.shape[0]):
    if idx_pt % 100 == 0:
        print(datetime.now().strftime("%H:%M:%S"), 'Computation of test data {}/{}.'.format(idx_pt+1, n_test))
    value = fun(X_test[idx_pt, :], A, BB, Rxx, T)
    list_values_test.append(value)

array_values_train = np.array(list_values_train)
array_values_test = np.array(list_values_test)


## Store values
np.save('X_train_finite_horizon.npy', X_train)
np.save('X_test_finite_horizon.npy', X_test)
np.save('array_values_train_finite_horizon.npy', array_values_train)
np.save('array_values_test_finite_horizon.npy', array_values_test)








# list_X = []

# array_t = np.linspace(0, 1, 30).reshape(1, -1)


# ## Sines with different frequencies, phases and offsets
# # for freq in 4 * np.random.rand(50) + 1:
# #     for phase in 2*math.pi * np.random.rand(50):
# #         for offset in [0]: # np.random.randn(10):
# #             f_func = lambda x: np.sin(freq*2*np.pi*x + phase) + offset

# #             X_ = f_func(array_t)
# #             X = X_ / np.max(np.abs(X_))

# #             list_X.append(X)

# ## Different polynomials
# # for idx0 in np.random.randn(5):
# #     for idx1 in np.random.randn(5):
# #         for idx2 in np.random.randn(5):
# #             for idx3 in np.random.randn(5):
# #                 for idx4 in np.random.randn(5):
# #                     f_func = lambda x: idx0 + idx1*x + idx2*x**2 + idx3*x**3 + idx4*x**4

# #                     X_ = f_func(array_t)
# #                     X = X_ / np.max(np.abs(X_))

# #                     list_X.append(X)

# ## Superconv of sines and cosines with different frequencs and fourier coeffs
# array_k = np.arange(1, 5).reshape(1, -1)


# list_array_a = [np.random.rand(1, 4)*2-1 for _ in range(200)]
# # list_array_a = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])]
# list_beta = [1.5, 2.0, 2.5, 3.0, 3.5]
# # list_beta = [3.0]
# array_scaling = [.01, .1, .2, .5, 1]

# for array_a in list_array_a:
#     for value_beta in list_beta:
#         for scaling in array_scaling:

#             X = scaling * array_a @ (array_k.T**(-value_beta) * np.cos(2 * math.pi * array_k.T * x.reshape(1, -1)))

#             list_X.append(X)

# # for freq in 4 * np.random.rand(50) + 1:
# #     for phase in 2*math.pi * np.random.rand(50):
# #         for offset in [0]: # np.random.randn(10):
# #             f_func = lambda x: np.sin(freq*2*np.pi*x + phase) + offset

# #             X_ = f_func(array_t)


# # find NaN values within X
# list_idx_keep = []
# for idx in range(len(list_X)):
#     if np.isnan(list_X[idx]).any():
#         print('NaN value found in list_X[{}]'.format(idx))
#     else:
#         list_idx_keep.append(idx)

# X0 = np.concatenate([list_X[idx] for idx in list_idx_keep], axis=0)
# X0 = X0 / 2
# X0[0, :] = 0        # include zero in training set!


# # # Compute inputs
# # value_beta = 3.0
# # array_a = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
# # X_test = np.zeros((array_a.shape[0], len(x)))

# # for idx_a in range(array_a.shape[0]):

# #     array_k = np.arange(1, 5).reshape(1, -1)
# #     X_test[idx_a, :] = (array_a[[idx_a], :]) @ (array_k.T**(-value_beta) * np.cos(2 * math.pi * array_k.T * x.reshape(1, -1)))

# # # normalize
# # X_test = X_test / np.max(np.abs(X_test))

# # X0 = np.concatenate([X0, X_test], axis=0)


# # # ALTERNATIVE TRAINING POINTS
# # N_train = 5000
# # sampler = qmc.Sobol(d=30, scramble=False)      # use low discrepancy points for increased stability
# # sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2))))-1
# # shuffle = np.random.permutation(sample.shape[0])

# # X0 = sample[shuffle[:N_train], :]
# # N_train = X0.shape[0]


# indices = np.arange(X0.shape[0])
# np.random.shuffle(indices)
# X0 = X0[indices, :]


# plt.figure(1)
# plt.clf()
# for idx in range(5):
#     plt.plot(array_t.reshape(-1), X0[idx, :].reshape(-1), 'x-')
# plt.show(block=False)




# # Compute value function

# list_values = [] 
# for idx_pt in range(X0.shape[0]):

#     value = v(X0[idx_pt,:].T).item()
    
#     list_values.append(value)


# array_values = np.array(list_values)




# # save array_values and X to file
# # np.save('array_values.npy', array_values)
# # np.save('X.npy', X0)

















