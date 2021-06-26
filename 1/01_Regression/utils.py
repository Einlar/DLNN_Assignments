import torch
import torch.nn as nn

from pathlib import Path
from datetime import datetime
import os

import logging

CHECKPOINT_PATH = Path("SavedModels")
DATE_FMT = "%d_%m_%y-%Hh%Mm%S"

def save_checkpoint(model : 'nn.Module',
                    optimizer : 'torch.optim',
                    nn_folder : str = '',
                    metadata : dict = None,
                    keep : int = -1):
    """
    Given a pytorch `model` and `optimizer`, saves all the parameters and the `metadata` in a .pth file named with the current date/time which is then stored inside a subfolder `nn_folder` placed inside CHECKPOINT_PATH. If `keep` > 0, only the most recent n files are kept (including the one just saved).
    """
    
    (CHECKPOINT_PATH / nn_folder).mkdir(parents=True, exist_ok=True) #Create folder for the network
    
    now = datetime.now()
    date = now.strftime(DATE_FMT) #Current time
    ext = ".pth"
    save_name = date + ext
    
    if metadata is None:
        metadata = dict()

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metadata
    }, (CHECKPOINT_PATH / nn_folder / save_name))
    
    if keep > 0:
        list_files = sorted((CHECKPOINT_PATH / nn_folder).glob("*.pth"), key=os.path.getmtime)
        
        for file in list_files[:-keep]: #Remove all other files
            file.unlink()

    if (CHECKPOINT_PATH / nn_folder / save_name).is_file():
        logging.info("Model successfully saved!")
        return True
    else:
        logging.error("Error in saving")
        return False

def list_checkpoints(nn_folder : str = ''):
    """
    List all the checkpoints inside `CHECKPOINT_PATH/nn_folder`
    """

    ext = ".pth"
    list_checkpoints = [f.stem for f in (CHECKPOINT_PATH / nn_folder).glob(f'*{ext}') if f.is_file()] 

    list_datetimes = [datetime.strptime(f, DATE_FMT) for f in list_checkpoints] #parse dates
    list_datetimes.sort() #sorts in-place


    return list_checkpoints


def load_checkpoint(nn_folder : str = '', version : int = -1): 
    """    
    Returns the dictionary stored in the @version-th .pth checkpoint inside CHECKPOINT_PATH/@nn_folder. 
    @version is an integer, with the oldest version denoted with 0, and the latest with -1 (as in an ordered list).
    """
    ext = ".pth"
    list_checkpoints = [f.stem for f in (CHECKPOINT_PATH / nn_folder).glob(f'*{ext}') if f.is_file()] 
    #Retrieves the list of all saved checkpoints, removing the .pth extension
    
    list_datetimes = [datetime.strptime(f, DATE_FMT) for f in list_checkpoints] #parse dates
    list_datetimes.sort() #sorts in-place
    
    if version < len(list_datetimes) and len(list_datetimes) > 0:
        filename = datetime.strftime(list_datetimes[version], DATE_FMT) + ext
        checkpoint = torch.load((CHECKPOINT_PATH / nn_folder / filename))
        logging.info(f"Checkpoint {filename} successfully loaded!")
        
        return checkpoint
    else:
        logging.error("File not found!")
        return None