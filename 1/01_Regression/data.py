import pandas as pd
import torch

import urllib.request
from pathlib import Path

from typing import Tuple
    
def download_dataset() -> Tuple['pd.DataFrame', 'pd.DataFrame']:
    """
    Download the dataset for the regression task, and return the train/test dataset as two Pandas DataFrames.
    """
    
    #---Download---#
    DATASET_PATH = Path("regression_dataset")

    if not DATASET_PATH.is_dir():
        DATASET_PATH.mkdir()

    dataset_filenames = {
        'train' : {
            "name" : "train_data.csv",
            "url"  : "https://gitlab.dei.unipd.it/gadaleta/nnld-2020-21-lab-resources/-/raw/master/homework_1_regression_dataset/train_data.csv"
        },
        'test' : {
            "name" : "test_data.csv",
            "url"  : "https://gitlab.dei.unipd.it/gadaleta/nnld-2020-21-lab-resources/-/raw/master/homework_1_regression_dataset/test_data.csv"
        }
    }

    for file in dataset_filenames.values():
        if not (DATASET_PATH / file['name']).is_file():
            print(f"{file['name']} missing!")

            urllib.request.urlretrieve(file['url'], DATASET_PATH / file['name'])

            if (DATASET_PATH / file['name']).is_file():
                print("Downloaded!\n")
    
    train_df = pd.read_csv(DATASET_PATH / dataset_filenames['train']['name'])
    test_df  = pd.read_csv(DATASET_PATH / dataset_filenames['test']['name'])
    
    return train_df, test_df
            
def PandasToTensor(df : pd.DataFrame,
                   features_col : list[str],
                   labels_col : list[str],
                   device : str = "cpu",
                   dtype : 'torch.dtype' = torch.float32) -> Tuple['torch.tensor', 'torch.tensor']:
    """
    Extract features and labels from specified columns of a Pandas DataFrame `df`, and return them as tensors of `dtype` on the `device`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be extracted
    features_col : list[str]
        Names of the columns of `df` that contain the *features* to be extracted
    labels_col : list[str]
        Names of the columns of `df` that contain the *labels* to be extracted
    device : torch.device
        Device to hold the returned tensors
    dtype : torch.dtype
        Dtype of the returned tensors
    
    Returns
    -------
    A tuple (features, labels) of torch.tensor of the specified `dtype` and `device`.
    """

    feat = torch.tensor(df[features_col].to_numpy(), dtype=dtype).to(device)
    lab  = torch.tensor(df[labels_col].to_numpy(), dtype=dtype).to(device)
    
    # if len(features_col) <= 1:
    #     feat = feat.unsqueeze(1) #Why this?
    # if len(labels_col) <= 1:
    #     lab  = lab.unsqueeze(1)
    return feat, lab 

