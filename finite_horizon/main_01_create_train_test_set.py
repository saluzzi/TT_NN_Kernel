# Create training and test data for the numerical experiments for Section 4.4.


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





