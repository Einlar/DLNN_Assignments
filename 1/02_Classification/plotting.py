import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from typing import Union

COLUMNWIDTH = 483.69 #page width of report .tex in pt

def show_histograms(model : 'nn.Model',
                    fig : 'plt.figure' = None,
                    **kwargs) -> None:
    """Plot the PDF of weights of all (Conv2D or Linear) layers of `model`. Additional `kwargs` are passed to the `plt.hist` function.

    Parameters
    ----------
    model : 'nn.Model'
        Instance of a Pytorch neural network model.
    fig : 'plt.figure'
        Matplotlib figure to be used for plotting. If None (default), a new figure is created.
    **kwargs : dictionary
        Optional arguments for `plt.hist`. 
    """
    MAX_PER_ROW = 2

    layers = [layer for layer in model.children()
                    if isinstance(layer, nn.Conv2d) or
                       isinstance(layer, nn.Linear)]

    n_layers = len(layers)

    n_cols = min(n_layers, MAX_PER_ROW)
    n_rows = math.ceil(n_layers / MAX_PER_ROW)

    conv_count = 0
    fc_count   = 0

    colors = get_cmap('Dark2').colors

    if fig is None:
        fig = plt.figure(figsize=(n_cols * 2.5, n_rows * 2.5))
        
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            name = 'conv' + str(conv_count)
            conv_count += 1
        else:
            name = 'fc' + str(fc_count)
            fc_count += 1
            
        weights = layer.weight.detach().cpu().numpy()

        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        ax.hist(weights.flatten(), bins=20, density=True, color=colors[i % len(colors)], **kwargs)
        ax.set_title(name)

    plt.tight_layout()
    plt.show()
    
def plot_conv_weights(model : 'nn.Model',
                      layer_id : int,
                      max_num : int = 0,
                      fig : 'plt.Figure' = None,
                      **kwargs) -> None:
    """Plot a maximum of `max_num` filters of a convolutional layer of `model` identified by `layer_id`. Additional `kwargs` can be passed to `plt.imshow`.

    Parameters
    ----------
    model : 'nn.Model'
        Instance of a Pytorch neural network model.
    layer_id : int
        Index of the convolutional layer to be plotted, following the order of `list(model.children())`.
    max_num : int, optional
        Maximum number of plotted filters, by default 0
    fig : 'plt.Figure'
        Matplotlib figure to be used for plotting. If None (default), a new figure is created.
    """
    MAX_PER_ROW = 8

    layer = list(model.children())[layer_id]

    #checking whether the layer is convolution layer or not 
    if isinstance(layer, nn.Conv2d):
        #getting the weight tensor data
        weights = layer.weight.detach().cpu().numpy()

        n_filters = weights.shape[0]

        if max_num > 0:
            n_filters = min(n_filters, max_num)

        n_rows = math.ceil(n_filters / MAX_PER_ROW)
        n_cols = min(n_filters, MAX_PER_ROW)

        if fig is None:
            fig = plt.figure(figsize=(n_cols, n_rows))

        for i in range(n_filters):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.imshow(weights[i, 0], **kwargs)
            
            ax.set_title(f"{i}")
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()
        plt.show()
        
    else:
        print("Not a convolutional layer")

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def show_sample(tensor : 'torch.tensor',
                label : Union[int, str] = None,
                predicted : Union[int, str] = None,
                ax : 'plt.ax' = None) -> None:
    """Prints the MNIST sample given by `tensor` (must be reshapable to a 28x28 image) on a given `ax` (created if None) along with the true `label` and the `predicted` one."""

    if ax is None:
        fig, ax = plt.subplots()

    img = tensor.cpu().numpy()

    assert np.prod(img.shape) == 28*28, "Wrong tensor dimensions"

    img = img.reshape(28, 28) * MNIST_STD + MNIST_MEAN #De-normalize

    ax.imshow(img, cmap='Greys')

    title = "Sample"
    if label is not None:
        title = f"True: {label}"
    if predicted is not None:
        title = f"Predicted: {predicted}"
    if label is not None and predicted is not None:
        title = f"y({label}) = {predicted}"

    ax.set_title(title)
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])