# THIS IS A FINAL FILE FOR THE KERNEL COMPUTATIONS WITHIN 4.3


import time

# Some imports
import numpy as np
from scipy.stats import qmc

from vkoga_2L import kernels
from low_versus_high_rank.Tizian.utils import SGD_done_right
from sklearn.neural_network import MLPRegressor
from datetime import datetime


np.random.seed(0)

# Settings      Create some data - do not use the matlab arrays
N_test = 10000      # increase to int(1e5)
N_train = 10000

C1, C2 = 1, 1
mu1, mu2 = 0, .5
sigma1, sigma2 = 1, 1

list_methods = ['1L'] # , 'NN']


loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)


array_dim = np.arange(3, 16+1)
# array_dim = np.arange(3, 5+1)

dic_results = {}
for method in list_methods:
    dic_results[method] = {'err_train': [], 'err_test': [], 'time_train': [], 'time_predict': []}


for dim in array_dim:

    print(datetime.now().strftime("%H:%M:%S"), 'dim = {}.'.format(dim))


    name_method = 'Mat2, 1/2, 10k'
    # kernel = kernels.Gaussian(ep=1/np.sqrt(dim))
    # kernel = kernels.Matern(k=4, ep=1/4 * 1/np.sqrt(dim))
    # kernel = kernels.Polynomial(a=2, p=7)
    # kernel = kernels.Gaussian(ep=2/np.sqrt(dim))
    kernel = kernels.Matern(k=2, ep=1/2 * 1/np.sqrt(dim))


    f_func = lambda x: (C1 * np.exp(-np.linalg.norm(x - mu1, axis=1)**2 / sigma1) \
        + C2 * np.exp(-np.linalg.norm(x - mu2, axis=1)**2 / sigma2)) * np.linalg.norm(x, axis=1)**2
    # f_func = lambda x: np.sum(x ** 2, axis=1, keepdims=True)
    

    # Get training and test set: [-1, 1]^d
    sampler = qmc.Sobol(d=dim, scramble=False)      # use low discrepancy points for increased stability
    sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2))))-1
    shuffle = np.random.permutation(sample.shape[0])

    # X_train = np.random.rand(N_train, dim)
    X_train = sample[shuffle[:N_train], :]
    X_train[0, :] = mu1
    X_train[1, :] = mu2

    X_test = 2*np.random.rand(N_test, X_train.shape[1]) - 1

    y_train = f_func(X_train)
    y_test = f_func(X_test)


    for method in list_methods:

        if method == '1L':
            # Compute kernel model and prediction
            t0_train_1L = time.time()
            A0 = kernel.eval(X_train, X_train) + 1e-8 * np.eye(N_train)
            coeff = np.linalg.solve(A0, y_train)
            t1_train_1L = time.time()

            # Compute errors
            t0_predict_1L = time.time()
            y_test_pred = kernel.eval(X_test, X_train) @ coeff
            t1_predict_1L = time.time()
            y_train_pred = A0 @ coeff


        elif method == 'NN':
            # regr = MLPRegressor(random_state=1, solver='lbfgs', hidden_layer_sizes=(100, 100, 100), max_iter=5000).fit(X_train, y_train.reshape(-1))
            t0_train_NN = time.time()
            regr = MLPRegressor(random_state=1, batch_size=64, hidden_layer_sizes=(100, 100), max_iter=5000)
            regr.fit(X_train, y_train.reshape(-1))
            t1_train_NN = time.time()

            
            t0_predict_1L = time.time()
            y_test_pred = regr.predict(X_test).reshape(-1)
            t1_predict_1L = time.time()
            y_train_pred = regr.predict(X_train).reshape(-1)


        res_train_1L = np.abs(y_train_pred.reshape(-1) - y_train.reshape(-1))
        res_test_1L = np.abs(y_test_pred.reshape(-1) - y_test.reshape(-1))

        
        # Compute errors
        # max_train, mse_train = np.max(res_train_1L), np.mean(res_train_1L**2) # OLD
        # max_test, mse_test = np.max(res_test_1L), np.mean(res_test_1L**2)     # OLD

        max_train = np.max(res_train_1L)
        err_train = loss_rel(y_train, y_train_pred)
        err_test = loss_rel(y_test, y_test_pred)

        dic_results[method]['err_train'].append(err_train)
        dic_results[method]['err_test'].append(err_test)
        dic_results[method]['time_train'].append(t1_train_1L - t0_train_1L)
        dic_results[method]['time_predict'].append(t1_predict_1L - t0_predict_1L)



from matplotlib import pyplot as plt


# check whether variable list_legend is defined
try:
    list_legend.append(name_method)
except NameError:
    list_legend = []


plt.figure(5)
plt.clf()
plt.plot(array_dim, dic_results['1L']['err_test'], 'x--')
# plt.plot(array_dim, dic_results['NN']['err_test'], 'x--')
plt.title('err_test')
plt.yscale('log')
plt.legend(list_legend)
plt.xlabel('dim')
plt.title(name_method)
plt.show(block=False)


# from matplotlib import pyplot as plt
# import numpy as np

# array_dim = np.array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])
# list_to_plot = [2.076804618434746e-05, 0.00017776878315597374, 0.00048791824809648205, 0.0019925183993045148, 0.00477994955953652, 0.011485151419467094, 0.02072287487044453, 0.034292352542609365, 0.057236650968846996, 0.07654083244509936, 0.1447941756809279, 0.15581610110893526, 0.20114200168123267, 0.19679899345898114]


# plt.figure(6)
# plt.clf()
# plt.plot(array_dim, list_to_plot, 'x--')
# # plt.plot(array_dim, dic_results['NN']['err_test'], 'x--')
# plt.title('err_test')
# plt.yscale('log')
# plt.legend('Mat2, 1/2, 10k')
# plt.xlabel('dim')
# plt.title(['Mat2, 1/2, 10k'])
# plt.show(block=False)


# import tikzplotlib
# tikzplotlib.save("accuracy_over_dimension.tex")