# asdf


# Some imports
import torch
import time
import numpy as np

import pytorch_lightning as pl
from scipy.stats import qmc
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from low_versus_high_rank.Tizian.utils import TorchDataset, NN


np.random.seed(0)

# Settings      Create some data - do not use the matlab arrays
case = 'case1'
N_train = 5000
N_test = 10000
dim = 16


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



loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)


# Load test set
sampler = qmc.Sobol(d=dim, scramble=False)      # use low discrepancy points for increased stability
sample = 2*sampler.random_base2(m=int(np.ceil(np.log(N_train) / np.log(2))))-1
shuffle = np.random.permutation(sample.shape[0])

# X_train = np.random.rand(N_train, dim)
X_train = sample[shuffle[:N_train], :]

X_test = 2*np.random.rand(N_test, X_train.shape[1]) - 1

y_train = f_func(X_train)
y_test = f_func(X_test)


# Compute NN
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
                            learning_rate=1e-3, decayRate=.9, decay_Epochs_nn=20).double()
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

# compute number of parameters of model_nn
num_params = 0
for param in model_nn.parameters():
    num_params += param.numel()
print('Number of parameters: ', num_params)


# Compute errors
max_train = np.max(res_train_1L)
err_train = loss_rel(y_train, y_train_pred)
err_test = loss_rel(y_test, y_test_pred)



print('NN training took {:.3f}s. NN prediction took {:.3f}s.'.format(t1_train - t0_train, t1_predict - t0_predict))
print('NN err_train = {:.3e}, err_test = {:.3e}'.format(err_train, err_test))


