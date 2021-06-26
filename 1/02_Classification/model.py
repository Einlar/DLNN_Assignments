import torch
import torch.nn as nn
class ConvNetMNIST(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.
    """

    def __init__(self, dropout_conv=0, activation_func='ReLU', **kwargs):
        super().__init__()

        #input.shape = (1, 28, 28) = (in_channels, in_height, in_width)
        
        #The output shape after a conv2d layer (or maxpool2d) is given by:
        #   out.shape = (out_channels, out_height, out_width) [see: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html]
        #where out_height = floor((in_height - kernel_height)/stride  + 1)
        #and similarly for out_width

        #input (1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) #(32, 24, 24)

        self.max_pool = nn.MaxPool2d(kernel_size=2) #(32, 12, 12)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) #(64, 8, 8)

        #maxpool => (64, 4, 4) => 64 * 4 * 4 = 1024

        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.act = getattr(nn, activation_func)()
        self.dropout_conv = nn.Dropout(dropout_conv)

    def forward(self, x : "torch.tensor"):
        x = self.conv1(x)
        x = self.act(x)
        
        x = self.max_pool(x)
        x = self.dropout_conv(x)
        
        x = self.conv2(x)
        x = self.act(x)
        
        x = self.max_pool(x)
        x = self.dropout_conv(x)

        x = x.view(-1, 1024)
    
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout_conv(x)
        
        x = self.fc2(x)

        return x