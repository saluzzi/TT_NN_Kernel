# Create training and test data for the numerical experiments for Section 4.5.



import math
import scipy
import numpy as np


from matplotlib import pyplot as plt
from scipy.stats import qmc

from Allen_Cahn.Tizian.utils import ValueFunction


desired_width=400 # 320
np.set_printoptions(linewidth=desired_width)


sigma = 1.e-2               # some parameter
N = 30                      # dimension
t0 = 0                      # initial time
t1 = 100                    # final time
dt = 0.001                  # temporal discretization step

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
P_sdre = lambda x: gamma*(scipy.linalg.sqrtm(dx*I/gamma+Ax(x)@Ax(x))+Ax(x))
v = lambda x: x.T @ P_sdre(x) @ x       # TODO: Approximate this function!!!



list_X = []

array_t = np.linspace(0, 1, 30).reshape(1, -1)


## Supercomp of sines and cosines with different frequencs and fourier coeffs
array_k = np.arange(1, 5).reshape(1, -1)


list_array_a = [np.random.rand(1, 4)*2-1 for _ in range(200)]
# list_array_a = [np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])]
list_beta = [1.5, 2.0, 2.5, 3.0, 3.5]
# list_beta = [3.0]
array_scaling = [.01, .1, .2, .5, 1]

for array_a in list_array_a:
    for value_beta in list_beta:
        for scaling in array_scaling:

            X = scaling * array_a @ (array_k.T**(-value_beta) * np.cos(2 * math.pi * array_k.T * x.reshape(1, -1)))

            list_X.append(X)


# find NaN values within X
list_idx_keep = []
for idx in range(len(list_X)):
    if np.isnan(list_X[idx]).any():
        print('NaN value found in list_X[{}]'.format(idx))
    else:
        list_idx_keep.append(idx)

X0 = np.concatenate([list_X[idx] for idx in list_idx_keep], axis=0)
X0 = X0 / 2
X0[0, :] = 0        # include zero in training set!


indices = np.arange(X0.shape[0])
np.random.shuffle(indices)
X0 = X0[indices, :]


plt.figure(1)
plt.clf()
for idx in range(5):
    plt.plot(array_t.reshape(-1), X0[idx, :].reshape(-1), 'x-')
plt.show(block=False)




# Compute value function
list_values = [] 
for idx_pt in range(X0.shape[0]):

    value = v(X0[idx_pt,:].T).item()
    
    list_values.append(value)

array_values = np.array(list_values)



# save array_values and X to file
np.save('array_values.npy', array_values)
np.save('X.npy', X0)

















