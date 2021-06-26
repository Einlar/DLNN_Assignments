import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

import time

import math

from utils import load_checkpoint, save_checkpoint
from typing import Tuple

from tqdm.auto import trange, tqdm
  
# if isnotebook():
#     from tqdm.notebook import trange, tqdm
# else:
#     from tqdm import trange, tqdm
    
#from tqdm.notebook import trange, tqdm

EVALUATION_PERIOD = 1
PATIENCE = 10

def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                l1_regularization : float = 0.,
                l2_regularization : float = 0.,
                progressbar = None,
                device = None
                ) -> float:
    """
    Train `model` (nn.Module instance) with a full pass over the dataset contained in `data_loader`, using `loss_fn`
    as loss function and the specified `optimizer`. 
    
    Returns
    -------
    average_train_loss, elapsed_time, average_train_accuracy
    """
    
    model.train()
    
    train_loss = 0.

    start = time.perf_counter()
    
    correct = 0.
    
    for (x_batch, y_batch) in data_loader:
        
        #iter_start = time.perf_counter()
        #Assume data is already on the correct device
        
        if device is not None:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        
        # Forward pass
        out = model(x_batch) 
        loss = loss_fn(out, y_batch) 

        #Apply regularization
        if l1_regularization > 0.: #L1
            for model_param_name, model_param_value in model.named_parameters():
                if model_param_name.endswith('weight'):
                    loss += l1_regularization * model_param_value.abs().sum()
        if l2_regularization > 0.: #L2
            for model_param_name, model_param_value in model.named_parameters():
                if model_param_name.endswith('weight'):
                    loss += l2_regularization * model_param_value.pow(2).sum()

        # Backward pass
        optimizer.zero_grad() 
        loss.backward()

        optimizer.step() #Update weights

        # Save loss
        train_loss += loss.detach().cpu().numpy()
        
        # Save accuracy
        correct += out.argmax(1).eq(y_batch).sum().detach().cpu().numpy()
        
        if progressbar is not None:
            progressbar.update()

    return train_loss / len(data_loader), time.perf_counter() - start, correct / (len(data_loader.dataset))


def val_epoch(model,
              data_loader,
              loss_fn,
              scheduler = None,
              device = None
              ) -> float:
    """
    Test `model` (nn.Module instance) with a full pass over the dataset contained in `data_loader`, using `loss_fn`
    as loss function. 
    
    Returns:
    average_val_loss
    """

    model.eval()

    val_loss = 0.
    correct = 0
    
    with torch.no_grad():
        for (x_batch, y_batch) in data_loader:
            if device is not None:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
            out = model(x_batch)
            loss = loss_fn(out, y_batch)

            val_loss += loss.detach().cpu().numpy()
            
            # Save accuracy
            correct += out.argmax(1).eq(y_batch).sum().detach().cpu().numpy()
            
        if scheduler is not None:
            scheduler.step()
    
    return val_loss / len(data_loader), correct / (len(data_loader.dataset))

def get_optimizer(net_params, hyperparams_learn):
    """
    Construct an optimizer as specified by the hyperparameters in `hyperparams_learn`, acting on the network parameters `net_params`.
    
    Returns
    -------
    optimizer
    """
    
    optim_choice = hyperparams_learn['optim']
    
    #Default parameters for each optimizer
    other_params = {}
    if optim_choice == 'SGD':
        other_params['momentum'] = 0.9
        other_params['nesterov'] = True
    
    optim = getattr(torch.optim, optim_choice)(net_params, lr=hyperparams_learn['learning_rate'], **other_params)
    return optim
    
def make_experiment(net_class, train, val,
                    hyperparams_net : dict = None,
                    hyperparams_learn : dict = None,
                    early_stopping : bool = False,
                    scheduler : bool = False,
                    scheduler_step_size : int = 50,
                    device : str = "cpu"):

    """
    Training and testing without cross-validation
    
    Parameters
    ----------
    net_class : nn.Module
        Class used to instantiate the network 
    train : torch.utils.data.Dataset
        Dataset for training
    val : torch.utils.data.Dataset
        Dataset for validation
    hyperparams_net : dict
        Dictionary used to instantiate the `net_class`, with `net_class(**hyperparams_net)`.
    hyperparams_learn: dict
        Dictionary with the hyperparameters used for learning. It must include the following keys:
            - "n_epochs": number of epochs for training
            - "batch_size"
            - "loss" : loss function to be used, picked from `torch.nn.{}` (e.g. if "loss": "MSELoss", the loss is picked from torch.nn.MSELoss)
            - "optim" : optimizer, picked from torch.optim.{}. If the `optim` is "SGD", then also "nesterov" (True/False) or "momentum" (float) can be specified as keys.
            - "learning_rate" : initial learning rate for the specified optimizer
    early_stopping : bool
        If True, training is stopped if the val_loss does not decrease during a number of epochs specified in the constant PATIENCE.
    scheduler : bool 
        If True, the learning_rate is decayed every n epochs [TODO Make it customizable]
    device : str
        Device to be used for training (e.g. "cpu"/"cuda"). 
            
    Returns
    -------
    trained_network : nn.Model 
        Network with the weights found by the training process
    (log_train_loss, log_val_loss) : (list, list) 
        Lists containing the value of the loss at each epoch (for the training) and at every EVALUATION_PERIOD epochs (for the validation).
    """
    
    required_learn_keys = ["n_epochs", "batch_size", "loss", "optim", "learning_rate"]
    
    for key in required_learn_keys:
        if key not in hyperparams_learn:
            raise KeyError(f"Missing key: {key} in hyperparams_learn")
    
    #hyperparams_net, hyperparams_learn = add_missing_hyperparams(user_hyperparams_net, user_hyperparams_learn)

    n_epochs = hyperparams_learn['n_epochs']
    batch_size = hyperparams_learn['batch_size']

    # train = TensorDataset(train_features, train_labels)
    # val   = TensorDataset(val_features,   val_labels)
        
    #Dataloaders
    train_loader  = DataLoader(train, batch_size = batch_size, shuffle = True, pin_memory=True)
    val_loader    = DataLoader(val,   batch_size = batch_size, shuffle = False, pin_memory=True)

    epoch_progressbar = trange(1, n_epochs + 1)

    net = net_class(**hyperparams_net) #Instantiate a new model & optimizer for each fold
    net.to(device)
    optim = get_optimizer(net.parameters(), hyperparams_learn)
    
    scheduler_fn = None
    
    if scheduler:
        scheduler_fn = torch.optim.lr_scheduler.StepLR(optim, step_size=scheduler_step_size, gamma=0.9) #one step = one batch
        
    loss_fn = getattr(nn, hyperparams_learn['loss'])()

    log_train_loss = np.zeros(n_epochs, dtype=float)
    log_val_loss   = np.zeros(n_epochs // EVALUATION_PERIOD, dtype=float)

    patience_counter = 0 #When this counter reaches PATIENCE, training is terminated

    min_val_loss = math.inf
    val_idx = 0

    iter_progressbar = trange(1, len(train_loader) + 1)
    for epoch in epoch_progressbar:
        iter_progressbar.reset()
        train_loss, time_per_epoch, train_acc = train_epoch(model=net, data_loader=train_loader, loss_fn=loss_fn, optimizer=optim, l1_regularization=hyperparams_learn.get('l1_regularization', 0.), l2_regularization=hyperparams_learn.get('l2_regularization', 0.), progressbar=iter_progressbar, device=device) #Train loop
        
        log_train_loss[epoch - 1] = train_loss
        last_train_loss = format(train_loss, '10.3f')

        if epoch % EVALUATION_PERIOD == EVALUATION_PERIOD-1:
            val_loss, val_acc = val_epoch(model=net, data_loader=val_loader, loss_fn=loss_fn, scheduler=scheduler_fn, device=device) #Evaluate on the validation dataset

            log_val_loss[val_idx,] = val_loss
            val_idx += 1 

            last_val_loss = format(val_loss, '10.3f')

            #Update info on the epoch progressbar
            desc = f"time/epoch: {time_per_epoch:10.0f}s, train: {last_train_loss}, {train_acc*100:.3f}%, val: {last_val_loss}, {val_acc*100:.3f}%"
            epoch_progressbar.set_description(desc)

            if early_stopping:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    epoch_progressbar.set_description(desc + " EarlySTOP")
                    break
    
    #Save checkpoint
    save_checkpoint(net, optim, 'experiments')

    return net, (log_train_loss, log_val_loss)

def evaluate(net, test_features, test_labels, loss_fn, batch_size = 1):
    """Evaluate a `net` on a test dataset"""
    
    test  = TensorDataset(test_features, test_labels)
    test_loader   = DataLoader(test,  batch_size = batch_size, shuffle = True)

    return val_epoch(model=net, data_loader=test_loader, loss_fn=loss_fn)


def predict(network : "nn.Model",
             dataset : "DataLoader",
             device  : str = "cpu") -> Tuple['np.ndarray', 'np.ndarray']:
    """
    Returns
    -------
    (predicted, true): (ndarray, ndarray)
        Predictions by the network and true labels as numpy arrays.
    """
    n_items = dataset.batch_size * len(dataset)
    predictions = np.zeros(n_items, dtype=int)
    labels      = np.zeros(n_items, dtype=int)

    network.eval()

    n_predicted = 0
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(dataset):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            out = network(x_batch)

            preds = out.argmax(1) #indices of maxima

            predictions[n_predicted : n_predicted + len(out)] = preds.detach().cpu().numpy().flatten()

            labels[n_predicted : n_predicted + len(out)] = y_batch.detach().cpu().numpy().flatten()

            n_predicted += len(out)
    return predictions, labels

def accuracy(network : "nn.Model",
             dataset : "DataLoader",
             device  : str = "cpu") -> float:
    """
    Accuracy [%] of `network` on `dataset` for a multi-class classification problem.
    """

    network.eval()

    correct = 0
    with torch.no_grad():
        for (x_batch, y_batch) in dataset:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            out = network(x_batch)

            predictions = out.argmax(1) #indices of maxima
            correct += predictions.eq(y_batch).sum()
    
    accuracy = correct / (len(dataset.dataset)) * 100.
        
    return float(accuracy.detach().cpu())