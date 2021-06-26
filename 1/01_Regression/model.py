import numpy as np
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    Simple Neural Network consisting of a customizable pile of Linear Layers.
    """

    def __init__(self,
                 input_dim : int = 1,
                 output_dim : int = 1,
                 hidden_neurons_per_layer : tuple = (10, 10),
                 activation_func : str = 'ReLU',
                 output_activation : bool = False,
                 dropout : float = -1,
                 verbose : bool = False,
                 **kwargs) -> None: 
        """
        Constructs a simple Multi Layer Perceptron with `input_dim` input units, a sequence of hidden layers with `hidden_neurons_per_layer` each, and `output_dim` output units. The activation function `activation_func` is applied to the output of each layer. If `output_activation` is False, no activation is applied to the very last layer.

        Parameters
        ----------
        input_dim : int
            Number of input neurons
        output_dim : int
            Number of output neurons
        hidden_neurons_per_layer : tuple
            `hidden_neurons_per_layer[i]` contains the number of neurons in the `i`-th hidden layer.
        activation_func : str
            Name of the activation function that is applied to the output of each layer. It is taken from the torch.nn module.
        output_activation : bool
            If `False`, the activation function is *not* applied to the final (output) layer
        dropout : float
            If it is a float in [0, 1), a Dropout layer with probability `dropout` is applied after every linear layer except the final one.
        verbose : bool
            If `True`, print debug messages.
        """
        
        super().__init__()
        
        self.arguments = {
            'input_dim' : input_dim,
            'output_dim' : output_dim,
            'hidden_neurons_per_layer' : hidden_neurons_per_layer,
            'activation_func' : activation_func,
            'output_activation' : output_activation,
            'verbose' : verbose
        }

        architecture = np.array([int(input_dim)] + \
                       list(hidden_neurons_per_layer) + \
                       [int(output_dim)], dtype=int) #Cast to int
        
        assert np.all(architecture > 0), "Input/output dimensionality and all layers' sizes must be positive integers!"
        
        
        self.num_layers = len(hidden_neurons_per_layer) + 1 # n. of hidden layers + output layer
        self.layers = nn.ModuleList([nn.Linear(in_features=architecture[i], out_features=architecture[i+1]) for i in range(self.num_layers)]) 
        #nn.ModuleList is needed to automatically add the nn.Linear parameters to the SimpleNet.
        #Otherwise, SimpleNet(...).parameters() would give an empty object
        
        self.act = getattr(nn, activation_func)()
        
        self.output_activation = output_activation

        if (dropout < 1.) and (dropout > 0.):
            self.dropout = True
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout = False
        
        if verbose:
            print(self)
    
    def __repr__(self):
        class_name = type(self).__name__

        arg_str = ', '.join([key + '=' + repr(val) for (key, val) in self.arguments.items()])
        
        return f'{class_name}({arg_str})'

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))

            if self.dropout:
                x = self.dropout_layer(x)
        
        #Last layer
        x = self.layers[-1](x) 
        
        if (self.output_activation):
            x = self.act(x)
        
        return x

