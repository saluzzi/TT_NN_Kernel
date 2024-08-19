import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch import nn


class SGD_done_right():
    # Implementation of "Stochastic Gradient Descent for GP done right" by JA Lin et al.,
    # https://openreview.net/forum?id=fj2E5OcLFn
    # https://anonymous.4open.science/r/SDD-GPs-5486/sdd.ipynb
    
    def __init__(self, kernel, reg_para=1e-4, lr=.001, momentum=.9, 
                 polyak=1e-2, iterations=30000, batch_size=10):

        self.kernel = kernel

        self.reg_para = reg_para                    # regularization parameter
        self.lr = lr                                # learning rate
        self.momentum = momentum                    # momentum
        self.polyak = polyak                        # polyak averaging
        self.iterations = iterations                # number of iterations
        self.batch_size = batch_size                # batch size

        # The following quantities will be set when calling .fit
        self.X_train = None
        self.y_train = None

        self.N = None

        

    # function to compute gradients
    def g(self, params, idx):

        K_batch = self.kernel(self.X_train[idx], self.X_train)

        grad = np.zeros((self.N,))
        grad[idx] = K_batch @ params - self.y_train[idx] + (self.reg_para ** 2) * params[idx]

        return (self.N / self.batch_size) * grad

    # function to perform one update step
    def update(self, params, params_polyak, velocity, idx):

        grad = self.g(params, idx)
        velocity = self.momentum * velocity - self.lr * grad
        params = params + velocity
        params_polyak = self.polyak * params + (1.0 - self.polyak) * params_polyak
        return params, params_polyak, velocity

    def fit(self, X_train, y_train):
        # Store datasets
        self.X_train = X_train      # these will be the "centers"
        self.y_train = y_train

        self.N = self.X_train.shape[0]

        # Initialize parameters
        alpha = np.zeros((self.N,))
        alpha_polyak = np.zeros((self.N,))
        v = np.zeros((self.N,))

        # Mini batch learning
        for idx_iteration in range(self.iterations):
            if idx_iteration % 1 == 0:
                print('Iteration: {}'.format(idx_iteration))

            idx = np.random.choice(self.N, size=(self.batch_size,), replace=False)
            alpha, alpha_polyak, v = self.update(alpha, alpha_polyak, v, idx)

        self.alpha_polyak = alpha_polyak

        return self
    
    def predict(self, x_pred):
        # make sure that method was fitted first!

        y_pred_sdd = self.kernel(x_pred, self.X_train) @ self.alpha_polyak

        return y_pred_sdd




class TorchDataset(Dataset):

    def __init__(self, data_input, data_output):
        self.data_input = data_input
        self.data_output = data_output

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return (self.data_input[idx], self.data_output[idx])




class Network(pl.LightningModule):
    """
    This class corresponds to an optimization of the model in a straightforward way.
    """

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        x_zeros = torch.zeros(1, x.shape[1]).double().requires_grad_(True)
        y_zeros = self(x_zeros)
        grad_y = torch.autograd.grad(outputs=y_zeros, inputs=x_zeros, 
                                     grad_outputs=torch.ones_like(y_zeros), create_graph=True)[0]
        
        loss = loss + torch.sum(grad_y**2)


        # Some logs
        self.log('train_loss', loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        # Some logs
        self.log('val_loss', loss, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)

        # Some logs
        self.log('test_loss', loss, prog_bar=True)

        return loss


class NN(Network):

    def __init__(self, dim_input=1, dim_output=1, width=32, learning_rate=1e-3, 
                 str_activ='ReLU', decayRate=.7, decay_Epochs_nn=10, flag_ResNet=False):
        super().__init__()

        # loss
        self.loss = nn.MSELoss()

        self.decayEpochs_nn = decay_Epochs_nn
        self.decayRate = decayRate

        self.flag_ResNet = flag_ResNet
        

        # Some settings
        self.learning_rate = learning_rate
        self.width = width
        self.dim_input = dim_input
        self.dim_output = dim_output

        # Define linear maps - hardcoded
        self.fc1 = nn.Linear(self.dim_input, self.width, bias=True)
        self.fc2 = nn.Linear(self.width, self.width, bias=True)
        self.fc3 = nn.Linear(self.width, self.width, bias=True)
        # self.fc4 = nn.Linear(self.width, self.width, bias=True)
        self.fc5 = nn.Linear(self.width, self.width, bias=True)
        self.fc6 = nn.Linear(self.width, dim_output, bias=True)

        # torch.nn.init.kaiming_normal_(self.fc1.weight)
        # torch.nn.init.kaiming_normal_(self.fc2.weight)
        # torch.nn.init.kaiming_normal_(self.fc3.weight)
        # torch.nn.init.kaiming_normal_(self.fc4.weight)
        # torch.nn.init.kaiming_normal_(self.fc5.weight)

        if str_activ == 'ReLU':
            self.activ_func = nn.ReLU()
        elif str_activ == 'Sigmoid':
            self.activ_func = nn.Sigmoid()
        elif str_activ == 'SELU':
            self.activ_func = nn.SELU()
        elif str_activ == 'Tanh':
            self.activ_func = nn.Tanh()
        elif str_activ == 'LeakyReLU':
            self.activ_func = nn.LeakyReLU()


    def forward(self, x):
        x = self.activ_func(self.fc1(x))

        if self.flag_ResNet:
            x = self.activ_func(self.fc2(x)) + x
            x = self.activ_func(self.fc3(x)) + x
            # x = self.activ_func(self.fc4(x) + x)
            x = self.activ_func(self.fc5(x)) + x
        else:
            x = self.activ_func(self.fc2(x))
            x = self.activ_func(self.fc3(x))
            # x = self.activ_func(self.fc4(x))
            x = self.activ_func(self.fc5(x))
        x = self.fc6(x)

        return x

    def configure_optimizers(self):  # Adam + LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # self.learning_rate)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.decayEpochs_nn, gamma=self.decayRate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, self.learning_rate)

        return [optimizer], [scheduler]


class NN_double(Network):

    def __init__(self, dim_input=1, dim_output=1, width=32, learning_rate=1e-3, 
                 str_activ='ReLU', decayRate=.7, decay_Epochs_nn=10, flag_ResNet=False):
        super().__init__()

        # loss
        self.loss = nn.MSELoss()

        self.decayEpochs_nn = decay_Epochs_nn
        self.decayRate = decayRate

        self.flag_ResNet = flag_ResNet
        

        # Some settings
        self.learning_rate = learning_rate
        self.width = width
        self.dim_input = dim_input
        self.dim_output = dim_output

        # Define linear maps - hardcoded
        self.fc1 = nn.Linear(self.dim_input, self.width, bias=True).double()
        self.fc2 = nn.Linear(self.width, self.width, bias=True).double()
        self.fc3 = nn.Linear(self.width, self.width, bias=True).double()
        # self.fc4 = nn.Linear(self.width, self.width, bias=True)
        self.fc5 = nn.Linear(self.width, self.width, bias=True).double()
        self.fc6 = nn.Linear(self.width, dim_output, bias=True).double()

        # torch.nn.init.kaiming_normal_(self.fc1.weight)
        # torch.nn.init.kaiming_normal_(self.fc2.weight)
        # torch.nn.init.kaiming_normal_(self.fc3.weight)
        # torch.nn.init.kaiming_normal_(self.fc4.weight)
        # torch.nn.init.kaiming_normal_(self.fc5.weight)

        if str_activ == 'ReLU':
            self.activ_func = nn.ReLU()
        elif str_activ == 'Sigmoid':
            self.activ_func = nn.Sigmoid()
        elif str_activ == 'SELU':
            self.activ_func = nn.SELU()
        elif str_activ == 'Tanh':
            self.activ_func = nn.Tanh()
        elif str_activ == 'LeakyReLU':
            self.activ_func = nn.LeakyReLU()


    def forward(self, x):
        x = self.activ_func(self.fc1(x))

        if self.flag_ResNet:
            x = self.activ_func(self.fc2(x)) + x
            x = self.activ_func(self.fc3(x)) + x
            # x = self.activ_func(self.fc4(x) + x)
            x = self.activ_func(self.fc5(x)) + x
        else:
            x = self.activ_func(self.fc2(x))
            x = self.activ_func(self.fc3(x))
            # x = self.activ_func(self.fc4(x))
            x = self.activ_func(self.fc5(x))
        x = self.fc6(x)

        return x

    def configure_optimizers(self):  # Adam + LR scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)  # self.learning_rate)

        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.decayEpochs_nn, gamma=self.decayRate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, self.learning_rate)

        return [optimizer], [scheduler]










