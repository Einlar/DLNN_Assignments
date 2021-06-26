import torch
import torch.nn as nn
import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib import ticker

import pandas as pd

from typing import Union

from mpl2latex import mpl2latex, latex_figsize

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

    ax.imshow(img, cmap='gist_gray')

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
    

#The code for the hovering images is taken from: https://stackoverflow.com/questions/42867400/python-show-image-upon-hovering-over-a-point
def plot_2D_representation_space(x, y, images, labels, fig = None):
    """
    Plot N points given their coordinates `x` and `y` in a 2D encoding space, coloring them according to their `label`, and showing 
    their respective `images` on mouse hover. 
    
    Parameters
    ----------
    x : "np.array" of shape (N,)
        Vector with values for the first encoded dimension
    y : "np.array" of shape (N,)
        Vector with values for the second encoded dimension
    images : "np.ndarray" of shape (N, width, height)
        `images[i]` will be shown when hovering the point at `(x[i], y[i])`
    labels : "np.ndarray" of shape (N,)
        Labels for the images (integer in [0, 9]).
    fig : "plt.figure"
        If provided, plot on this figure. Otherwise, a new figure is generated.
    """

    with mpl2latex(False):
        # create figure and plot scatter
        if fig is None:
            fig = plt.figure(figsize=latex_figsize(wf=.9, columnwidth=COLUMNWIDTH))

        ax = fig.add_subplot(111)

        colors = get_cmap('tab10').colors
        point_colors = np.array(colors)[labels]
        line = ax.scatter(x, y, marker="o", c=point_colors)

        # create the annotations box
        im = OffsetImage(images[0,:,:], zoom=3, cmap='Greys')
        xybox=(50., 50.)
        ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
                boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
        # add it to the axes and make it invisible
        ax.add_artist(ab)
        ab.set_visible(False)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'{i}',
                        markerfacecolor=colors[i]) for i in range(10)]
        ax.legend(handles=legend_elements, ncol=5, columnspacing=.5, handletextpad=-.1) #handletextpad = spacing between each marker and its label

        def hover(event):
            # if the mouse is over the scatter points
            if line.contains(event)[0]:
                # find out the index within the array from the event
                ind, = line.contains(event)[1]["ind"]
                # get the figure size
                w,h = fig.get_size_inches()*fig.dpi
                ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
                hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
                # if event occurs in the top or right quadrant of the figure,
                # change the annotation box position relative to mouse.
                ab.xybox = (xybox[0]*ws, xybox[1]*hs)
                # make annotation box visible
                ab.set_visible(True)
                # place it at the position of the hovered scatter point
                ab.xy =(x[ind], y[ind])
                # set the image corresponding to that point
                im.set_data(images[ind,:,:])
            else:
                #if the mouse is not over a scatter point
                ab.set_visible(False)
            fig.canvas.draw_idle()

            # add callback for mouse moves

        fig.canvas.mpl_connect('motion_notify_event', hover)       

        ax.set_xlabel('Enc. Variable 0')    
        ax.set_ylabel('Enc. Variable 1')
        ax.set_title("Internal Representation of Digits")

        #plt.savefig("Plots/2d_representation.pdf", transparent=True, bbox_inches='tight')

        #plt.show()

def plot_reconstruction_error(metrics : "pd.DataFrame", ax : "plt.ax" = None, title = "CNN AutoEncoder Learning Curve"):
    if ax is None:
        fig, ax = plt.subplots(figsize=latex_figsize(wf=.9, columnwidth=COLUMNWIDTH))

    xs = np.arange(len(metrics) - 1) + 1

    ax.plot(xs, metrics['train_loss'][1:], c='k', label="Train")
    ax.plot(xs, metrics['val_loss'][1:], c='r', label="Validation")

    ax.set_title(title)

    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_xlabel("Epoch")
    ax.legend()

    formatter = ticker.ScalarFormatter(useMathText=True) #Put the multiplier (e.g. *1e-2) at the top left side, above the plot, making everything more "compact"
    formatter.set_scientific(True) 
    formatter.set_powerlimits((-1,1)) 


    ax.yaxis.set_major_formatter(formatter)


    ax.patch.set_facecolor('white')
    plt.tight_layout()