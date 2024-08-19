# Numerical experiments for Section 4.3 using NNs.


import time

# Some imports
import numpy as np
from scipy.stats import qmc

from sklearn.neural_network import MLPRegressor
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from low_versus_high_rank.Tizian.utils import TorchDataset, NN
from torch.utils.data import DataLoader




np.random.seed(0)

# Settings      Create some data - do not use the matlab arrays
N_test = 10000      # increase to int(1e5)
N_train = 10000

C1, C2 = 1, 1
mu1, mu2 = 0, .5
sigma1, sigma2 = 1, 1

list_methods = ['NN']


loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)


array_dim = np.arange(3, 16+1)
# array_dim = np.arange(3, 5+1)

dic_results = {}
for method in list_methods:
    dic_results[method] = {'err_train': [], 'err_test': [], 'time_train': [], 'time_predict': []}


for dim in [14, 15, 16]: #array_dim:

    print(datetime.now().strftime("%H:%M:%S"), 'dim = {}.'.format(dim))

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

    y_train = f_func(X_train).reshape(-1, 1)
    y_test = f_func(X_test).reshape(-1, 1)


    for method in list_methods:

        if method == '1L':
            pass


        elif method == 'NN':
            

            # Set up data for NN training
            N_data = X_train.shape[0]
            fraction_train = .9

            train_dataset = TorchDataset(data_input=torch.from_numpy(X_train[:int(fraction_train*N_data), :]),
                                        data_output=torch.from_numpy(y_train[:int(fraction_train*N_data)]))
            val_dataset = TorchDataset(data_input=torch.from_numpy(X_train[int(fraction_train*N_data):, :]),
                                    data_output=torch.from_numpy(y_train[int(fraction_train*N_data):]))

            train_loader = DataLoader(train_dataset, batch_size=128, num_workers=1, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, num_workers=1, shuffle=False)



            ## Compute NN model
            model_nn = NN(dim_input=X_train.shape[1], dim_output=1, width=512, str_activ='ReLU',
                                        learning_rate=5e-3, decayRate=.9, decay_Epochs_nn=20).double()
            t0_train = time.time()
            trainer = pl.Trainer(callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=10)], 
                                    max_epochs=100)
            trainer.fit(model_nn, train_dataloaders=train_loader, val_dataloaders=val_loader)
            t1_train = time.time()

        y_train_pred = model_nn(torch.from_numpy(X_train)).detach().numpy()
        t0_predict = time.time()
        y_test_pred = model_nn(torch.from_numpy(X_test)).detach().numpy()
        t1_predict = time.time()

        res_train_1L = np.abs(y_train_pred.reshape(-1) - y_train.reshape(-1))
        res_test_1L = np.abs(y_test_pred.reshape(-1) - y_test.reshape(-1))

        
        # Compute errors
        max_train = np.max(res_train_1L)
        err_train = loss_rel(y_train, y_train_pred)
        err_test = loss_rel(y_test, y_test_pred)

        dic_results[method]['err_train'].append(err_train)
        dic_results[method]['err_test'].append(err_test)
        dic_results[method]['time_train'].append(t1_train - t0_train)
        dic_results[method]['time_predict'].append(t1_predict - t0_predict)







