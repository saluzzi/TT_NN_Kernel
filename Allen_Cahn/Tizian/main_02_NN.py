# THIS IS A FINAL FILE FOR THE NN COMPUTATIONS WITHIN 4.4



import math
import numpy as np
import scipy
import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from low_versus_high_rank.Tizian.utils import TorchDataset, NN
from torch.utils.data import DataLoader

from Allen_Cahn.Tizian.utils import ValueFunction
import time
from matplotlib import pyplot as plt


desired_width=400 # 320
np.set_printoptions(linewidth=desired_width)


## Load dataset
array_X = np.load('X_FINAL.npy')
array_values = np.load('array_values_FINAL.npy').reshape(-1, 1)

X_train = array_X
y_train = array_values


## Settings
# General settings
dim = 30


## Set up data for NN training
N_data = X_train.shape[0]
fraction_train = .9

train_dataset = TorchDataset(data_input=torch.from_numpy(X_train[:int(fraction_train*N_data), :]),
                            data_output=torch.from_numpy(y_train[:int(fraction_train*N_data), :]))
val_dataset = TorchDataset(data_input=torch.from_numpy(X_train[int(fraction_train*N_data):, :]),
                        data_output=torch.from_numpy(y_train[int(fraction_train*N_data):, :]))

train_loader = DataLoader(train_dataset, batch_size=128, num_workers=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=1, shuffle=False)



## Compute NN model
model_nn = NN(dim_input=X_train.shape[1], dim_output=y_train.shape[1], width=512, str_activ='ReLU',
                            learning_rate=5e-3, decayRate=.9, decay_Epochs_nn=20).double()
t0_train = time.time()
trainer = pl.Trainer(callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=30)], 
                        max_epochs=5)
trainer.fit(model_nn, train_dataloaders=train_loader, val_dataloaders=val_loader)
t1_train = time.time()

# save checkpoint
# trainer.save_checkpoint('Allen_Cahn/Tizian/checkpoints/final_Allen_Cahn_NN.ckpt')

# model_numpy = lambda x: kernel.eval(x, X_train) @ coeff

# Compute training loss
loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)

train_loss = loss_rel(y_train, model_nn(torch.from_numpy(X_train)).detach().numpy())
print('err_train: {:.3e}'.format(train_loss))           # train + validation

train_loss = loss_rel(y_train[:int(fraction_train*N_data)], model_nn(torch.from_numpy(X_train[:int(fraction_train*N_data)])).detach().numpy())
print('err_train: {:.3e}'.format(train_loss))           # train



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
    v_surr = model_nn(torch.from_numpy(X_test[[idx], :])).detach().numpy().item()

    print('{:.4f}, {:.4f}, absolute diff: {:.4e}, err_test = {:.3e}'.format(
        v_true, v_surr, v_true - v_surr, loss_rel(v_true, v_surr)))

    print(' ')


    # Compute value function
    valFunc_sdre = ValueFunction(t0, t1, dt, dx, gamma, lambda x: P_sdre(x), lambda x: Ax(x), str_control='sdre')
    total_cost_sdre, list_y_sdre = valFunc_sdre.value_function_mathias(X_test[idx,:])

    # Compute value function using the NN surrogate
    # deactivate gradients of model_nn
    model_nn.eval()

    # compute gradients of model_nn with respect to input
    def gradient_of_nn(x):
        assert len(x) == X_train.shape[1], 'Something is weird'

        x_tensor = torch.from_numpy(x).reshape(1, -1).double()
        x_tensor.requires_grad = True
        y_tensor = model_nn(x_tensor)
        y_tensor.backward()
        return x_tensor.grad.detach().numpy().reshape(X_train.shape[1], 1)  # I want (30, 1) output




    valFunc_surr = ValueFunction(t0, t1, dt, dx, gamma, lambda x: P_sdre(x), lambda x: Ax(x), 
                                lambda x: gradient_of_nn(x), str_control='surr')
    total_cost_surr, list_y_surr = valFunc_surr.value_function_mathias(X_test[idx,:])

    print('cost sdre = {:.4f}, cost surr = {:.4f}, err_cost = {:.4f}'.format(
        total_cost_sdre, total_cost_surr, np.abs(total_cost_sdre - total_cost_surr)))




## Some visualization
plt.figure(10000)
plt.clf()
for idx in np.linspace(0, 600, 10):
    idx = int(idx)
    plt.plot(list_y_sdre[idx], 'x-', label='time={:.3f}'.format(idx*dt))
plt.legend()
plt.title('SDRE')
plt.show(block=False)

plt.figure(10001)
plt.clf()
for idx in np.linspace(0, 600, 10):
    idx = int(idx)
    plt.plot(list_y_surr[idx], 'x-', label='time={:.3f}'.format(idx*dt))
plt.legend()
plt.title('SURR')
plt.show(block=False)



array_sdre = np.stack(list_y_sdre)
array_surr = np.stack(list_y_surr)



# store array_sdre as matlab
import scipy.io
scipy.io.savemat('array_sdre.mat', {'array_sdre': array_sdre})
scipy.io.savemat('array_surr.mat', {'array_surr': array_surr})













x, y = next(iter(train_loader))


y_pred = model_nn(x)
loss = model_nn.loss(y_pred, y)

x_zeros = torch.zeros(1, 30).double().requires_grad_(True)
y_zeros = model_nn(x_zeros)
grad_y = torch.autograd.grad(outputs=y_zeros, inputs=x_zeros, 
                                grad_outputs=torch.ones_like(y_zeros), create_graph=True)[0]

print(grad_y.shape)




