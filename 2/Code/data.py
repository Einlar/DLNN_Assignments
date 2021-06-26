import torch
from torch.utils.data import DataLoader, random_split, Subset

import torchvision
from torchvision.datasets import MNIST

import pytorch_lightning as pl

from sklearn.model_selection import KFold

from typing import Tuple, Optional

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

class FastMNIST(MNIST): 
    """
    Improves performance of training on MNIST by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    
    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #"Transparent" interface with MNIST class

        #Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255) #Divide by 255

        #Normalize it with the usual MNIST mean and std [REMOVED]
        #self.data = self.data.sub_(MNIST_MEAN).div_(MNIST_STD) #the underscore denotes in-place operations 

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

        return img, target
    
class MapDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset : "torch.utils.data.Dataset",
                 transform = None):
        """
        Given a dataset of tuples (features, labels),
        returns a new dataset with a transform applied to the features (lazily, only when an item is called).

        Note that data is not cloned/copied from the initial dataset.
        
        Parameters
        ----------
        dataset : "torch.utils.data.Dataset"
            Dataset of tuples (features, labels)
        transform : function
            Transformation applied to the features of the original dataset
        """
    
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
    
class NoisyDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset : "torch.utils.data.Dataset",
                 noise_transform : "torchvision.transforms"):
        """
        Given a `dataset`, generate new "noisy" samples by appling a given `noise_transform`.
        Items are returned as a tuple of 3 elements: (features_clean, features_noisy, label)
        
        Parameters
        ----------
        dataset : "torch.utils.data.Dataset"
            Dataset of tuples (features, labels)
        noise_transform : "torchvision.transforms"
            Transformation that add noise to the features from the dataset.
        """

        self.dataset = dataset
        self.noise_transform = noise_transform

    def __getitem__(self, index) -> Tuple["torch.tensor", "torch.tensor", "torch.tensor"]:
        x_clean, y = self.dataset[index][0], self.dataset[index][1]
        x_noisy = self.noise_transform(x_clean)
    
        return x_clean, x_noisy, y

    def __len__(self):
        return len(self.dataset) 
    
class AddGaussianNoise():
    def __init__(self,
                 mean : float = 0.,
                 std : float = 1.,
                 clip : bool = True):
        """
        Transformation that adds random gaussian noise to a tensor:
        new_tensor = original_tensor + random_normal(mean, std)
        
        Parameters
        ----------
        mean : float, optional
            mean of normal distribution, by default 0.
        std : float, optional
            standard deviation of normal distribution, by default 1.
        clip : bool, optional
            If True, after adding the noise, values are clipped to the [0, 1] range.
        """
        self.std = std
        self.mean = mean
        self.clip = clip
    
    def __call__(self, tensor : 'torch.tensor'):
        new_tensor = tensor + torch.randn(tensor.size(), device=tensor.device) * (self.std ** .5) + self.mean
        if self.clip:
            return torch.clamp(new_tensor, min=0., max=1.)
        else:
            return new_tensor
        
class AddSaltPepperNoise(): #Adapted from skimage.random_noise from scikit-learn (https://scikit-image.org/docs/dev/api/skimage.util.html#random-noise)

    def __init__(self,
                 amount : float = .05,
                 salt_vs_pepper : float = 0.5,
                 clip : bool = True):
        """
        Transformation that adds "Salt & Pepper" noise to a tensor. 
        Any element (chosen randomly with prob. `amount`) is set to `1` (salt, with probability `salt_vs_pepper`) or to `0` (pepper). 
        
        Parameters
        ----------
        amount : float, optional
            Probability that an entry of tensor will be affected by the noise. 
        salt_vs_pepper : float, optional
            (Conditional) probability that an entry that is affected by noise will be set to `1` (salt) rather than `0` (pepper).
        clip : bool, optional
            If True, after adding the noise, values are clipped to the [0, 1] range.
        """
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper
        self.clip = clip
    
    def __call__(self, tensor : 'torch.tensor'):
        affected_by_noise = torch.rand_like(tensor) < self.amount
        salted = torch.rand_like(tensor) < self.salt_vs_pepper
        peppered = ~salted

        new_tensor = torch.clone(tensor)
        new_tensor[affected_by_noise & salted] = 1
        new_tensor[affected_by_noise & peppered] = 0

        if self.clip:
            return torch.clamp(new_tensor, min=0., max=1.)
        else:
            return new_tensor
        
class AddSpeckleNoise(): #Adapted from skimage.random_noise from scikit-learn

    def __init__(self,
                 mean : float = 0.,
                 std : float = 1.,
                 clip : bool = True):
        """
        Transformation that adds "Speckle" noise to a tensor:
        new_tensor = tensor + tensor * random_gaussian(mean, std)
        
        Parameters
        ----------
        mean : float, optional
            mean of normal distribution, by default 0.
        std : float, optional
            standard deviation of normal distribution, by default 1.
        clip : bool, optional
            If True, after adding the noise, values are clipped to the [0, 1] range.
        """
        self.std = std
        self.mean = mean
        self.clip = clip
    
    def __call__(self, tensor : 'torch.tensor'):
        new_tensor = tensor + tensor * (torch.randn(tensor.size(), device=tensor.device) * (self.std ** .5) + self.mean)
        if self.clip:
            return torch.clamp(new_tensor, min=0., max=1.)
        else:
            return new_tensor
class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, batch_size : int = 64, train_data_transform : "torchvision.transforms" = None):
        super().__init__()

        self.batch_size = batch_size
        self.train_data_transform = train_data_transform
    
    def setup(self, stage: Optional[str] = None):
        self.test_dataset = FastMNIST('classifier_data', train=False, download=True)

        self.full_train_dataset = FastMNIST('classifier_data', train=True, download=True)
        
        #Train-Validation split
        n_samples = len(self.full_train_dataset)
        n_training = int(0.8 * n_samples) #80%-20% split

        self.train_dataset, self.val_dataset = random_split(self.full_train_dataset, [n_training, n_samples-n_training])

        if self.train_data_transform is not None:
            self.train_dataset = MapDataset(self.train_dataset, self.train_data_transform) #Apply transform just to the training part of the data


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

class MNIST_CV():

    def __init__(self,
                 kfolds : int = 4,
                 batch_size : int = 64,
                 train_data_transform : "torchvision.transforms" = None):
        """
        Splits training data into folds for cross-validation, returning DataLoaders for each train/val split.
        
        Parameters
        ----------
        kfolds : int = 4
            Number of folds for cross-validation
        batch_size : int = 64
            Size of batches returned by the DataLoaders
        train_data_transform : "torchvision.transforms" = None
            Transformation to be applied to the train split of each fold    
        """

        self.batch_size = batch_size
        self.train_data_transform = train_data_transform
        self.full_train_dataset = FastMNIST('classifier_data', train=True, download=True)

        self.kf = KFold(n_splits = kfolds, shuffle=True)

    
    def get_fold(self) -> Tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
        """Yields a tuple of DataLoaders for the train/validation split of each fold
        
        Returns
        -------
        A generator of tuples (train_dataloader, val_dataloader)
        """

        for (train_indices, val_indices) in self.kf.split(self.full_train_dataset):
            train_dataset = Subset(self.full_train_dataset, train_indices)

            if self.train_data_transform is not None:
                train_dataset = MapDataset(train_dataset, self.train_data_transform) #Apply transform just to the training part

            val_dataset = Subset(self.full_train_dataset, val_indices)

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader   = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            yield (train_dataloader, val_dataloader)

class MNISTNoisy(pl.LightningDataModule):
    def __init__(self,
                 noise_transform : "torchvision.transforms",
                 batch_size : int = 64,
                 train_data_transform : "torchvision.transforms" = None,
                 val_percentage_size : float = 20.):
        """
        MNIST Dataset for training a Denoising AutoEncoder within the Pytorch-Lightning framework.

        Parameters
        ----------
        noise_transform : "torchvision.transforms"
            Transformation applied to all datasets (train/val/test) to add noise.
            For each sample, both a "clean" and a "noisy" version are stored in the dataset, along with its label.
        batch_size : int = 64
            Batch size of all datasets 
        train_data_transform : "torchvision.transforms"
            Transforms that are applied only to the train dataset (not val, not test) BEFORE adding the noise. They can be used, for example, for dynamic data augmentation (i.e. generation of new samples during each epoch by adding random transformations to the original training dataset).
        val_percentage_size : float = 20.
            Percentage of all training samples that are added to the validation dataset. By default, val_percentage_size=20. denotes a 80%-20% split between training/validation.
        """
        
        super().__init__()

        self.batch_size = batch_size
        self.train_data_transform = train_data_transform
        self.noise_transform = noise_transform
        self.val_percentage_size = val_percentage_size
        
    
    def setup(self, stage: Optional[str] = None):
        """Initialize train/val/test dataset and apply transforms"""

        self.test_dataset = FastMNIST('classifier_data', train=False, download=True)

        self.full_train_dataset = FastMNIST('classifier_data', train=True, download=True)
        
        #Train-Validation split
        n_samples = len(self.full_train_dataset)
        n_training = int( (1. - self.val_percentage_size/100) * n_samples)

        self.train_dataset, self.val_dataset = random_split(self.full_train_dataset, [n_training, n_samples-n_training])

        if self.train_data_transform is not None:
            self.train_dataset = MapDataset(self.train_dataset, self.train_data_transform) #Apply transform just to the training part of the data

        #Apply noise
        self.train_dataset = NoisyDataset(self.train_dataset, self.noise_transform)
        self.val_dataset   = NoisyDataset(self.val_dataset, self.noise_transform)
        self.test_dataset  = NoisyDataset(self.test_dataset, self.noise_transform)
    
    #---Create DataLoaders---#
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
