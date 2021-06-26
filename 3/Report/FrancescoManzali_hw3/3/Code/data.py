import numpy as np
from collections import deque
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from typing import Tuple, Iterator

class ReplayMemory():
    def __init__(self, capacity : int):
        """
        Collection of replays for training a RL agent,
        stored as a deque in the `memory` attribute.

        Parameters
        ----------
        capacity : int 
            Maximum number of replays that can be stored at once.
            When capacity is full, the oldest replays are removed
            to make space for the new ones.
        """

        self.memory = deque(maxlen=capacity)

    def push(self,
             state : "np.ndarray",
             action : int, #Should an int be used here?
             next_state : "np.ndarray",
             reward : float,
             done : bool = False,
             verbose : bool = False) -> None:
        """
        Append the results from env.step to the memory.

        Parameters
        ----------
        state : "np.ndarray"
            Array of floats representing the current state of the environment
            (before the action is made).
        action : int
            Action to be performed
        next_state : "np.ndarray"
            Array of floats representing the state after the action is made
            (i.e. the newly "observed" state)
        reward : float
            Reward received after performing `action` at `state`
        done : bool = False
            If True, then the observed `next_state` is a final state for
            the environment, and no more actions are allowed.
        verbose : bool = False
            If True, print debug messages.
        """

        self.memory.append( (state, action, next_state, reward, done) )  

        if verbose:
            print(f"Adding: ({state}, {action}, {next_state}, {reward}, {done})")

    def sample(self, n_samples : int = 1) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
        """
        Return a batch of distinct samples from the memory.

        Parameters
        ----------
        n_samples : int = 1
            Number of samples to be returned.

        Returns
        -------
        Tuple[states, actions, next_states, rewards, dones]

        states : np.ndarray, shape: (n_samples, state_dim), dtype: np.float32
        actions : np.ndarray, shape: (n_samples,), dtype: np.int64
        next_states : np.ndarray, shape: (n_samples, state_dim), dtype: np.float32
        rewards : np.ndarray, shape: (n_samples,), dtype: np.float32
        dones : np.ndarray, shape: (n_samples,), dtype: bool
        """

        assert n_samples <= len(self.memory), f"Requested {n_samples} samples, but the memory contains only {len(self.memory)} of them."

        indices = np.random.choice(len(self.memory), n_samples, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[idx] for idx in indices])

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool)
        ) 

    def __len__(self):
        return len(self.memory)

class MemoryDataset(IterableDataset):
    """
    Wrapper for ReplayMemory, allowing PyTorch to use it as an (Iterable)Dataset.
    """

    def __init__(self, buffer : ReplayMemory, epoch_size : int = 1000) -> None:
        """
        Converts a ReplayMemory object to an IterableDataset that can be used by PyTorch.

        Parameters
        ----------
        buffer : ReplayMemory
            Memory of replays to be used as training dataset.
        epoch_size : int = 1000
            Amount of frames that are loaded within an epoch.
            It is also the maximum batch_size obtainable when accessing this dataset
            through a PyTorch DataLoader.
        """

        self.buffer = buffer
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]]:
        """
        Yields samples from the ReplayMemory.
        """

        states, actions, next_states, rewards, dones = self.buffer.sample(self.epoch_size)

        #Better to return them one-by-one with a generator, so that PyTorch can optimize the transfer to the GPU
        for i in range(len(dones)):
            yield states[i], actions[i], next_states[i], rewards[i], dones[i]
    
    def __len__(self):
        return self.epoch_size

#states, actions, next_states, rewards, dones = batch

def test_ReplayMemory():
    # Define the replay memory
    print("Initializing ReplayMemory(capacity=3)")
    replay_mem = ReplayMemory(capacity=3)

    # Push some samples
    assert len(replay_mem) == 0
    replay_mem.push(1,1,1,1, verbose=True)
    assert len(replay_mem) == 1

    replay_mem.push(2,2,2,2, verbose=True)
    assert len(replay_mem) == 2

    replay_mem.push(3,3,3,3, verbose=True)
    assert len(replay_mem) == 3
    
    replay_mem.push(4,4,4,4, verbose=True)
    assert len(replay_mem) == 3
    
    replay_mem.push(5,5,5,5, verbose=True)
    assert len(replay_mem) == 3
    

    # Check the content of the memory
    assert (1, 1, 1, 1, False) not in replay_mem.memory
    assert (2, 2, 2, 2, False) not in replay_mem.memory
    assert (3, 3, 3, 3, False) in replay_mem.memory
    assert (4, 4, 4, 4, False) in replay_mem.memory
    assert (5, 5, 5, 5, False) in replay_mem.memory

    # Random sample
    sample = replay_mem.sample(2)

    for s in sample:
        assert len(s) == 2

def test_MemoryDataset():
    replay_mem = ReplayMemory(capacity=5)

    replay_mem.push(1,1,1,1)
    replay_mem.push(2,2,2,2)
    replay_mem.push(3,3,3,3)
    replay_mem.push(4,4,4,4)
    replay_mem.push(5,5,5,5)

    mem_dataset = MemoryDataset(replay_mem, 3)

    num = 0
    for sample in mem_dataset:
        assert sample in replay_mem.memory
        num += 1
    
    assert num == 3

def test_dataloader():
    #---Test with a dataloader---#

    replay_mem = ReplayMemory(capacity=3)

    replay_mem.push(1,1,1,1)
    replay_mem.push(2,2,2,2)
    replay_mem.push(3,3,3,3)

    memo_dataset = MemoryDataset(replay_mem, 3)

    dataloader = DataLoader(dataset = memo_dataset, sampler=None, batch_size=1)

    #Fill with new samples
    replay_mem.push(4,4,4,4)
    replay_mem.push(5,5,5,5)
    replay_mem.push(6,6,6,6)

    #Test that the dataloader is updated
    #(i.e. if changes to replay_mem are relayed to the dataloader)
    for sample in dataloader: #batch_size = 1 means that single samples are returned
        assert tuple(sample) in replay_mem.memory


