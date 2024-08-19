


import time
import numpy as np
from scipy.stats import qmc
import datetime

from vkoga import kernels



np.random.seed(0)

# Settings      Create some data - do not use the matlab arrays
case = 'reg_1'
N_test = 10000
dim = 16

loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)


# Define the different cases
if case == 'a':
    flag_02 = False
    f_func = lambda x: np.exp(-np.sum(x, axis=1, keepdims=True) / (2*dim))
elif case == 'b':
    flag_02 = False
    f_func = lambda x: np.exp(-np.prod(x, axis=1, keepdims=True))
elif case == 'c':
    flag_02 = True
    f_func = lambda x: np.exp(-np.prod(x, axis=1, keepdims=True))
elif 'reg' in case:
    flag_02 = False
    dic_para = {'reg_1': [1, 0, 0], 'reg_2': [1, .5, 0], 
                'reg_3': [1, .5, .5], 'reg_4': [0, .5, 0], 'reg_5': [0, 0, 0.5]}
    
    f_func = lambda x: dic_para[case][0] * np.sum(x ** 2, axis=1, keepdims=True) \
                   + dic_para[case][1] * np.linalg.norm(x - .5, axis=1, keepdims=True) \
                   + dic_para[case][2] * np.sqrt(np.linalg.norm(x - (-.5), axis=1, keepdims=True))
    

# Define stuff we want to loop over
list_kernels = [kernels.Matern(k=0), kernels.Matern(k=2), kernels.Matern(k=4),
                kernels.Gaussian(), kernels.Polynomial(a=2, p=7), kernels.Polynomial(a=1, p=5)]
list_shape_para = [.05/np.sqrt(dim), .1/np.sqrt(dim), .2/np.sqrt(dim), .5/np.sqrt(dim), 
                   1/np.sqrt(dim), 2/np.sqrt(dim)]
list_reg_para = [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
list_n_train = [5000, 2000, 1000]

list_kernels = [kernels.Matern(k=2)]
list_shape_para = [.5/np.sqrt(dim), .25/np.sqrt(dim), .125/np.sqrt(dim)]
list_reg_para = [0]
list_n_train = [5000]

# list_kernels = [kernels.Gaussian()]
# list_shape_para = [.1/np.sqrt(dim)]
# list_reg_para = [0]
# list_n_train = [998]

N_cases = (len(list_kernels)-2) * len(list_shape_para) * len(list_reg_para) * len(list_n_train) + 2 * len(list_reg_para) * len(list_n_train)


# Load test set
if flag_02:
    X_test = 2*np.random.rand(N_test, dim)
else:
    X_test = 2*np.random.rand(N_test, dim) - 1
y_test = f_func(X_test)


# Loop over all cases
idx_counter = 0

dic_results = {}

print(case)
for N_train in list_n_train:

    dic_results[N_train] = {}

    # Compute training inputs
    sampler = qmc.Sobol(d=dim, scramble=False)      # use low discrepancy points for increased stability
    if flag_02:
        sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2))))
    else:
        sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2)))) - 1
    shuffle = np.random.permutation(sample.shape[0])

    X_train = sample[shuffle[:N_train], :]
    N_train = X_train.shape[0]

    # Compute training targets
    y_train = f_func(X_train)  

    for kernel in list_kernels:
        dic_results[N_train][kernel.name] = {}


        for idx_shape_para, shape_para in enumerate(list_shape_para):
            dic_results[N_train][kernel.name][shape_para] = {}

            if 'polynomial' in kernel.name and idx_shape_para == 0:
                pass        # polynomial kernel have no shape parameter
            elif 'polynomial' in kernel.name:
                continue
            else:
                kernel.set_params(shape_para)


            for reg_para in list_reg_para:

                # Compute model
                t0_train_1L = time.time()
                A0 = kernel.eval(X_train, X_train) + reg_para * np.eye(N_train)
                coeff = np.linalg.solve(A0, y_train)
                t1_train_1L = time.time()

                # Compute training and test prediction
                y_train_pred = kernel.eval(X_train, X_train) @ coeff
                t0_test_1L = time.time()
                y_test_pred = kernel.eval(X_test, X_train) @ coeff
                t1_test_1L = time.time()

                res_train_1L = np.abs(A0 @ coeff - y_train)
                res_test_1L = np.abs(y_test_pred - y_test)

                # Compute errors
                # max_train, err_train = np.max(res_train_1L), np.mean(res_train_1L**2)
                # err_test = np.mean(res_test_1L**2)

                max_train = np.max(res_train_1L)
                err_train = loss_rel(y_train, y_train_pred)
                err_test = loss_rel(y_test, y_test_pred)

                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                      ': {}/{}: err_train = {:.3e}, err_test = {:.3e}.     training time = {:.3f}s, test_time = {:.3f}s. N_train = {}, kernel = {}, shape_para = {}, reg_para = {}'.format(
                          idx_counter, N_cases, err_train, err_test, t1_train_1L-t0_train_1L, t1_test_1L-t0_test_1L, N_train, kernel.name, shape_para, reg_para))

                idx_counter += 1

                # Save results
                dic_results[N_train][kernel.name][shape_para][reg_para] = {'max_train': max_train, 'err_train': err_train, 'err_test': err_test}

                if idx_counter % 10 == 0 or idx_counter == N_cases-1:
                    np.save('results_' + case + '_{}.npy'.format(idx_counter), dic_results)








