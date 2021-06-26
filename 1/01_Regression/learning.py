import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import KFold

import time

import math

from data import PandasToTensor
from utils import load_checkpoint, save_checkpoint

from tqdm.auto import tqdm, trange

EVALUATION_PERIOD = 1
PATIENCE = 10

def train_epoch(model,
                data_loader,
                loss_fn,
                optimizer,
                l1_regularization : float = 0.,
                l2_regularization : float = 0.) -> float:
    """
    Train `model` (nn.Module instance) with a full pass over the dataset contained in `data_loader`, using `loss_fn`
    as loss function and the specified `optimizer`. 
    
    Returns
    -------
    average_train_loss, elapsed_time
    """
    
    model.train()
    
    train_loss = 0.

    start = time.perf_counter()

    for (x_batch, y_batch) in data_loader:
        #Assume data is already on the correct device

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

    return train_loss / len(data_loader), time.perf_counter() - start


def val_epoch(model,
              data_loader,
              loss_fn,
              scheduler = None) -> float:
    """
    Test `model` (nn.Module instance) with a full pass over the dataset contained in `data_loader`, using `loss_fn`
    as loss function. 
    
    Returns:
    average_val_loss
    """

    model.eval()

    val_loss = 0.

    with torch.no_grad():
        for (x_batch, y_batch) in data_loader:
            out = model(x_batch)
            loss = loss_fn(out, y_batch)

            val_loss += loss.detach().cpu().numpy()
            
        if scheduler is not None:
            scheduler.step()
    
    return val_loss / len(data_loader)


def cross_validate(features : 'torch.tensor',
                   labels : 'torch.tensor',
                   batch_size : int = 10,
                   folds : int = 5,
                   random_state : int = 1):
    """
    Generate DataLoader of train/validation datasets for cross validation.
    
    The dataset specified by (`features`, `labels`) is split into a number of folds given by `folds`. 
    Then samples are batched by DataLoaders according to `batch_size`. 
    """

    kf = KFold(n_splits = folds, random_state = random_state, shuffle = True)

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(features)):
        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx],   labels[val_idx]

        train = TensorDataset(x_train, y_train)
        val   = TensorDataset(x_val, y_val)
        
        #Dataloaders
        train_loader = DataLoader(train, batch_size = batch_size, shuffle = True)
        val_loader   = DataLoader(val,   batch_size = batch_size, shuffle = True)

        yield (train_loader, val_loader)


def add_missing_hyperparams(user_hyperparams_net : dict = None, user_hyperparams_learn : dict = None):
    """ 
    Adds missing hyperparameters to the dictionaries provided by the user, initializing them to default values.
    (hyperparams_net: hyperparameters for instantiating the network, hyperparams_learn: hyperparameters for learning).
    
    Returns
    -------
    (hyperparams_net, hyperparams_learn) with the missing keys added
    """
    
    hyperparams_net = {
            'input_dim'  : 1,
            'output_dim' : 1,
            'hidden_neurons_per_layer' : (10,10,10,10),
            'activation_func' : 'ReLU',
            'output_activation' : False,
            'verbose' : False
    }

    hyperparams_learn = {
        'batch_size' : 7,
        'learning_rate' : 0.003,
        'n_epochs' : 500,
        'l1_regularization' : 0.,
        'l2_regularization' : 0.,
        'optim' : 'Adam',
        'loss' : 'MSELoss',
        'folds' : 5,
        'epoch_progressbar' : False
    }

    #Update with the (optional) hyperparameters provided as arguments
    if user_hyperparams_net is not None:
        hyperparams_net.update(user_hyperparams_net)
    if user_hyperparams_learn is not None:
        hyperparams_learn.update(user_hyperparams_learn)

    return hyperparams_net, hyperparams_learn

def make_experiment_cv(net_class, features_train, labels_train,
                       user_hyperparams_net : dict = None,
                       user_hyperparams_learn : dict = None,
                       device : str = "cpu",
                       reload : bool = True,
                       scheduler : bool = False):

    """Evaluate the given hyperparameters with cross-validation.
    
    First, the network is instantiated by `net_class`, using the hyperparameters from `user_hyperparams_net`.
    Then, the dataset specified in (`features_train`, `labels_train`) is used to form train/val splits
    according to the number of folds specified in `user_hyperparams_learn`, and the model is trained over them.
    
    If `reload` is True, the returned network will have the parameters from the best performing fold (the one with the lowest val_loss). 
    Otherwise, the parameters from the last trained fold are kept.
    
    If `scheduler` is True, the learning rate is exponentially decayed during training.
    
    Returns
    -------
    (network, (log_train_loss, log_val_loss))
    """

    hyperparams_net, hyperparams_learn = add_missing_hyperparams(user_hyperparams_net, user_hyperparams_learn)

    n_epochs = hyperparams_learn['n_epochs']
    n_folds   = hyperparams_learn['folds']

    loss_fn = getattr(nn, hyperparams_learn['loss'])()

    log_train_loss = np.zeros((n_epochs, n_folds), dtype=float)
    log_val_loss   = np.zeros((n_epochs, n_folds), dtype=float) 

    folds = tqdm(cross_validate(features_train, labels_train, batch_size=hyperparams_learn['batch_size'], folds=n_folds),
            desc=f"Cross-validation folds",
            total=n_folds)   

    for fold_id, (train_loader, val_loader) in enumerate(folds): #Cross-validation loop

        last_train_loss = 'NaN'
        last_val_loss = 'NaN'

        net = net_class(**hyperparams_net) #Instantiate a new model & optimizer for each fold
        net.to(device)
        optim = get_optimizer(net.parameters(), hyperparams_learn)
        
        scheduler_fn = None
        if scheduler:
            scheduler_fn = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)

        if hyperparams_learn['epoch_progressbar']:
            epoch_iterator = trange(1, n_epochs + 1)
        else:
            epoch_iterator = range(1, n_epochs + 1)

        val_idx = 0 #Number of times that the validation set was used

        for epoch in epoch_iterator:
            train_loss, time_per_epoch = train_epoch(model=net, data_loader=train_loader, loss_fn=loss_fn, optimizer=optim, l1_regularization=hyperparams_learn.get('l1_regularization', 0.), l2_regularization=hyperparams_learn.get('l2_regularization', 0.)) #Train loop

            log_train_loss[epoch - 1, fold_id] = train_loss
            last_train_loss = format(train_loss, '10.3f')

            if epoch % EVALUATION_PERIOD == EVALUATION_PERIOD-1:
                val_loss = val_epoch(model=net, data_loader=val_loader, loss_fn=loss_fn, scheduler=scheduler_fn) #Evaluate on the validation dataset

                log_val_loss[val_idx, fold_id] = val_loss
                val_idx += 1 

                last_val_loss = format(val_loss, '10.3f')

            
            if hyperparams_learn['epoch_progressbar']:
                #Update info on the epoch progressbar
                desc = f"time/epoch: {time_per_epoch*1000:10.3f}ms, loss (train): {last_train_loss}, loss (val): {last_val_loss}"
                epoch_iterator.set_description(desc)

        fold_val_last    = log_val_loss[-1, fold_id]
        
        train_losses = log_train_loss[-1, :fold_id+1] #The losses accumulated from cross validation until now
        val_losses   = log_val_loss[-1, :fold_id+1]

        #Update info on the folds progressbar
        desc = f"Cross-validation folds: [Train] last: {train_losses[-1]:.3f}, min: {np.min(train_losses):.3f}#{np.argmin(train_losses)} "
        desc += f"[Val] last: {val_losses[-1]:.3f}, min: {np.min(val_losses):.3f}#{np.argmin(val_losses)}"
        folds.set_description(desc)

        if fold_val_last < min(log_val_loss[-1, :fold_id], default=math.inf): #If the last iteration has a lower val_loss than any other iteration
            save_checkpoint(net, optim, 'cv', keep=1) #Save checkpoint if it has the best cross-validation
            

    if reload:
        checkpoint = load_checkpoint('cv') #Load last checkpoint in this folder
        net.load_state_dict(checkpoint['model_state_dict'])
    #optim.load_state_dict(checkpoint['optimizer_state_dict'])

    return net, (log_train_loss, log_val_loss)

def get_optimizer(net_params, hyperparams_learn : dict):
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
    
def make_experiment(net_class, train_features, train_labels, val_features, val_labels,
                    user_hyperparams_net : dict = None,
                    user_hyperparams_learn : dict = None,
                    early_stopping : bool = False,
                    scheduler : bool = False,
                    device : str = "cpu"):

    """Training and testing without cross-validation.
       
       Instantiates a network from `net_class`, using the specified hyperparameters in `user_hyperparams_net`. 
       Then trains it, according to the hyperparameters in `user_hyperparams_learn`, on the training dataset given 
       by (`train_features`, `train_labels`), validating on the 
       dataset given by (`val_features`, `val_labels`).
       
       Optionally, training can be stopped if the validation loss does not improve in PATIENCE epoch by setting `early_stopping` to True.
       The learning rate is exponentially decayed during training if `scheduler` is True.
       
       Returns
       -------
       network, (log_train_loss, log_val_loss)
       """

    hyperparams_net, hyperparams_learn = add_missing_hyperparams(user_hyperparams_net, user_hyperparams_learn)

    n_epochs = hyperparams_learn['n_epochs']
    batch_size = hyperparams_learn['batch_size']

    train = TensorDataset(train_features, train_labels)
    val   = TensorDataset(val_features,   val_labels)
        
    #Dataloaders
    train_loader  = DataLoader(train, batch_size = batch_size, shuffle = True)
    val_loader    = DataLoader(val,   batch_size = batch_size, shuffle = False)

    epoch_progressbar = trange(1, n_epochs + 1)

    net = net_class(**hyperparams_net) #Instantiate a new model & optimizer for each fold
    net.to(device)
    optim = get_optimizer(net.parameters(), hyperparams_learn)
    
    scheduler_fn = None
    
    if scheduler:
        scheduler_fn = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        
    loss_fn = getattr(nn, hyperparams_learn['loss'])()

    log_train_loss = np.zeros(n_epochs, dtype=float)
    log_val_loss   = np.zeros(n_epochs // EVALUATION_PERIOD, dtype=float)

    patience_counter = 0 #When this counter reaches PATIENCE, training is terminated

    min_val_loss = math.inf
    val_idx = 0

    for epoch in epoch_progressbar:
        train_loss, time_per_epoch = train_epoch(model=net, data_loader=train_loader, loss_fn=loss_fn, optimizer=optim, l1_regularization=hyperparams_learn.get('l1_regularization', 0.), l2_regularization=hyperparams_learn.get('l2_regularization', 0.)) #Train loop

        log_train_loss[epoch - 1] = train_loss
        last_train_loss = format(train_loss, '10.3f')

        if epoch % EVALUATION_PERIOD == EVALUATION_PERIOD-1:
            val_loss = val_epoch(model=net, data_loader=val_loader, loss_fn=loss_fn, scheduler=scheduler_fn) #Evaluate on the validation dataset

            log_val_loss[val_idx,] = val_loss
            val_idx += 1 

            last_val_loss = format(val_loss, '10.3f')

            #Update info on the epoch progressbar
            desc = f"time/epoch: {time_per_epoch*1000:10.3f}ms, loss (train): {last_train_loss}, loss (val): {last_val_loss}"
            epoch_progressbar.set_description(desc)

            if early_stopping:
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE:
                    epoch_progressbar.set_postfix("EarlySTOP")
                    break
    
    #Save checkpoint
    save_checkpoint(net, optim, 'experiments')

    return net, (log_train_loss, log_val_loss)

def evaluate(net, test_features, test_labels, loss_fn, batch_size = 1):
    """Evaluate a `net` on a test dataset"""
    
    test  = TensorDataset(test_features, test_labels)
    test_loader   = DataLoader(test,  batch_size = batch_size, shuffle = True)

    return val_epoch(model=net, data_loader=test_loader, loss_fn=loss_fn)
