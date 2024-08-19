# Numerical experiments for Section 4.4 using NNs.

import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from low_versus_high_rank.Tizian.utils import TorchDataset, NN
from torch.utils.data import DataLoader

import time



desired_width=400 # 320
np.set_printoptions(linewidth=desired_width)


## Load dataset
X_train = np.load('X_train_finite_horizon.npy')
X_test = np.load('X_test_finite_horizon.npy')
y_train = np.load('array_values_train_finite_horizon.npy').reshape(-1, 1)
y_test = np.load('array_values_test_finite_horizon.npy').reshape(-1, 1)


## Settings
# General settings
dim = 30


## Set up data for NN training
N_data = X_train.shape[0]
fraction_train = .9

train_dataset = TorchDataset(data_input=torch.from_numpy(X_train[:int(fraction_train*N_data), :]),
                            data_output=torch.from_numpy(y_train[:int(fraction_train*N_data)]))
val_dataset = TorchDataset(data_input=torch.from_numpy(X_train[int(fraction_train*N_data):, :]),
                        data_output=torch.from_numpy(y_train[int(fraction_train*N_data):]))

train_loader = DataLoader(train_dataset, batch_size=128, num_workers=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, num_workers=1, shuffle=False)



## Compute NN model
model_nn = NN(dim_input=X_train.shape[1], dim_output=y_train.shape[1], width=512, str_activ='ReLU',
                            learning_rate=5e-3, decayRate=.9, decay_Epochs_nn=20).double()
t0_train = time.time()
trainer = pl.Trainer(callbacks = [EarlyStopping(monitor="val_loss", mode="min", patience=30)], 
                        max_epochs=100)
trainer.fit(model_nn, train_dataloaders=train_loader, val_dataloaders=val_loader)
t1_train = time.time()

# save checkpoint
# trainer.save_checkpoint('Allen_Cahn/Tizian/checkpoints/final_Allen_Cahn_NN.ckpt')


# Compute training loss
loss_rel = lambda y_true, y_pred: np.sqrt(np.linalg.norm(y_true - y_pred)**2 / np.linalg.norm(y_true)**2)

t0 = time.time()
y_train_pred = model_nn(torch.from_numpy(X_train)).detach().numpy()
t1 = time.time()
y_test_pred = model_nn(torch.from_numpy(X_test)).detach().numpy()
t2 = time.time()


train_loss = loss_rel(y_train, y_train_pred)
test_loss = loss_rel(y_test, y_test_pred)



print('Train loss: {:.3e}. Training prediction for {} points took {:.3e}s. Training took {:.3e}s.'.format(
    train_loss, X_train.shape[0], t1-t0, t1_train-t0_train))
print('Test loss:  {:.3e}. Test prediction for {} points took {:.3e}s.'.format(test_loss, X_test.shape[0], t2-t1))







