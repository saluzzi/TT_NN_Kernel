# Numerical experiments for Section 4.2 using kernel methods.


# Some imports
import time
import numpy as np
from scipy.stats import qmc
from vkoga import kernels

np.random.seed(0)

# Settings      Create some data - do not use the matlab arrays
case = 'case1'
N_train = 2000
N_test = 10000
dim = 16
# kernel = kernels.Gaussian(ep=.1/np.sqrt(dim))
kernel = kernels.Matern(k=2, ep=.01/np.sqrt(dim))
# kernel = kernels.Polynomial(a=1, p=5)

if case == 'case1':
    lambda0, lambda1, lambda2 = 1, 0, 0
elif case == 'case2':
    lambda0, lambda1, lambda2 = 1, 0.5, 0
elif case == 'case3':
    lambda0, lambda1, lambda2 = 1, 0.5, 0.5
elif case == 'case4':
    lambda0, lambda1, lambda2 = 0, 0.5, 0
elif case == 'case5':
    lambda0, lambda1, lambda2 = 0, 0, .5

f_func = lambda x: lambda0 * np.sum(x ** 2, axis=1, keepdims=True) \
                   + lambda1 * np.linalg.norm(x - .5, axis=1, keepdims=True) \
                   + lambda2 * np.sqrt(np.linalg.norm(x - (-.5), axis=1, keepdims=True))




# Load test set
sampler = qmc.Sobol(d=dim, scramble=False)      # use low discrepancy points for increased stability
sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2))))-1
shuffle = np.random.permutation(sample.shape[0])

# X_train = np.random.rand(N_train, dim)
X_train = sample[shuffle[:N_train], :]

X_test = 2*np.random.rand(N_test, X_train.shape[1]) - 1

y_train = f_func(X_train)
y_test = f_func(X_test)


# Compute kernel model and prediction
t0_train_1L = time.time()
A0 = kernel.eval(X_train, X_train)
coeff = np.linalg.solve(A0, y_train)
t1_train_1L = time.time()


# Compute errors
t0_predict_1L = time.time()
y_test_pred = kernel.eval(X_test, X_train) @ coeff
t1_predict_1L = time.time()

res_train_1L = np.abs(A0 @ coeff - y_train)
res_test_1L = np.abs(y_test_pred - y_test)
max_train, mse_train = np.max(res_train_1L), np.mean(res_train_1L**2)
max_test, mse_test = np.max(res_test_1L), np.mean(res_test_1L**2)
# err_test_2 = norm(Vx_test-Value_test(:,1))/norm(Value_test(:,1));

# The following is the test error which Luca always considers
mse_test = np.linalg.norm(y_test_pred - y_test) / np.linalg.norm(y_test_pred)



print('1L training took {:.3f}s. 1L prediction took {:.3f}s.'.format(t1_train_1L - t0_train_1L, t1_predict_1L - t0_predict_1L))
print('Train: max error = {:.3e}. mse error = {:.3e}. sqrt of mse error = {:.3e}.'.format(max_train, mse_train, np.sqrt(mse_train)))
print('Test: max error =  {:.3e}. mse error = {:.3e}. sqrt of mse error = {:.3e}.'.format(max_test, mse_test, np.sqrt(mse_test)))
print(' ')
print('Values for table:')
print('{:.2e}, {:.2e}, {:.2e}'.format(max_train, np.sqrt(mse_train), np.sqrt(mse_test)))



