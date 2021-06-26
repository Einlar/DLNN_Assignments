import torch
from torchvision.datasets import MNIST

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081
COLUMNWIDTH = 483.69

class FastMNIST(MNIST): 
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    
    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #"Transparent" interface with MNIST class

        #Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255) #Divide by 255

        #Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(MNIST_MEAN).div_(MNIST_STD) #the underscore denotes in-place operations 

        #Put both data and targets on GPU in advance
        #device = ...
        #self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index : int):
        """ 
        Parameters
        ----------
        index : int
            Index of the element to be returned
        
        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """

        img = self.data[index]
        target = self.targets[index]
        
        #To apply transformations "on the fly" for dynamic data augmentation (each epoch has a "different version" of the dataset):
        # if self.transform is not None:
        #     img = self.transform(img)

        # target = self.targets[index]

        return img, target
    
class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a transformation
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:     
            x = self.transform(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
            
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset) 
    
class AddGaussianNoise():
    """Transformation that adds random gaussian noise to a tensor"""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor : 'torch.tensor'):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean