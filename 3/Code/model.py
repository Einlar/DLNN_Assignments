import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import numpy as np

import gym
from utils import wrap_env
from data import ReplayMemory, MemoryDataset
from agent import Agent

import logging
from typing import Tuple, Callable, Union

from tqdm.notebook import trange


class DQN(nn.Module):
    """Fully-connected policy network for RL on simple gym environments"""

    def __init__(self,
                 state_space_dim : Tuple[int],
                 action_space_dim : int):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(state_space_dim[0], 128),
            nn.Tanh(),
            nn.Linear(128, action_space_dim)
        )

    def forward(self, x : "torch.tensor"):
        return self.linear(x)

class DeepQLearner(pl.LightningModule):

    def __init__(self,
                 env : Union[str, "gym.Env"],
                 Network : "nn.Module",
                 hyper_parameters : dict = None,
                 save_video_every_n_episodes : int = 100,
                 add_to_reward : Callable[["np.ndarray", float], float] = None,
                 chosen_policy : str = "softmax",
                 update_temp_every_frame : bool = False,
                 update_target_every_frame : bool = False,
                 count_steps : bool = True,
                 preprocess_env = None,
                 **kwargs) -> None:
        """
        Pytorch Lightning module for Reinforcement Learning on simple gym environments.

        Parameters
        ----------
        env : Union[str, "gym.Env"]
            Name of the gym environment (e.g. "CartPole-v1"),
            or directly an instance of a gym environment.
            Note: due to a bug of gym, classic control environments
            (such as CartPole) cannot be pickled if env.render()
            has been called (https://github.com/rlworkgroup/garage/issues/292),
            so they cannot be passed directly. In this case, pass the string
            instead, and use `preprocess_env` to specify the
            wrapping transformation.
        Network : nn.Module
            Class used to instantiate the policy network. 
            It is instantiated with the required input/output dimensions as arguments:
            ```net = Network(state_space_dim, action_space_dim)```
        hyper_parameters : dict = None
            Hyperparameters for the RL task. 
            If provided, all of the hyperparameters *must* be specified.
            Otherwise, default values are used, which can be overridden
            by passing other values as keyword arguments.

            They are as follows:
            - "gamma": discount factor for future rewards
            - "replay_memory_capacity": how many frames can be stored at once
            - "batch_size" : size of batches
            - "learning_rate": initial learning rate for the SGD optimizer (with no momentum)
            - "target_net_update_steps": to stabilize training, a "target network"
               is used to fix an objective for the loss, and it is updated to 
               the parameters of the policy network every this many episodes/frames,
               depending on the value of `target_net_update_steps` (see Double Deep Q-Learning)
            - "min_samples_for_training": amount of frames to be generated before
               training starts.
            - "loss_function": loss 
            - "steps_per_epoch": how many frames are cycled through per epoch.
               Note that an epoch is just a fixed amount of frames, it does not 
               necessarily coincide with an episode.
            - "initial_temperature": starting temperature for the softmax/epsilon-greedy Agent policy
            - "reach_zero_temperature_after_n_episodes": 
               duration of exponential decay of temperature before 0 is reached.
               Temperature is updated after each full episode.
            - "temperature_policy" (optional): array of temperatures to be used
               instead of a standard exponential decay from `initial_temperature`.
        save_video_every_n_episodes : int = 100
            Frequency of saving videos of episodes to the `video` folder.
        add_to_reward : function(state, reward) -> new_reward
            If specified, it is used to add a custom reward to the one
            returned by the environment.
        chosen_policy : str = "softmax"
            Policy for the Agent. Either "softmax" or "epsilon-greedy". 
        update_temp_every_frame : bool = False
            If True, temperature is updated at every frame.
            Otherwise, it is updated only when an episode ends.
        update_target_every_frame : bool = False
            If True, the target policy network is updated at
            every `target_net_update_steps` frames.
            Otherwise, `target_net_update_steps` specifies a number of 
            episodes, not frames.
        count_steps : bool = True
            If True, the returned reward is equal to the number of steps for
            which the agent "survives" in the environment.
            Otherwise, the actual accumulated rewards from the environment
            are returned.
        **kwargs
            Any hyperparameter can be passed as a keyword argument,
            and will override the default values (or the ones provided through
            the dictionary `hyper_parameters`)

        """ 

        super().__init__()
        
        if hyper_parameters is None:
            hyper_parameters = {
                "gamma" : .97, 
                "replay_memory_capacity" : 10000, 
                "batch_size" : 128,
                "learning_rate" : 1e-2,
                "target_net_update_steps" : 10, #Number of episodes to wait before updating the target network
                "min_samples_for_training" : 2048, #Minimum samples in the replay memory to enable the training
                "loss_function" : "SmoothL1Loss",
                "steps_per_epoch" : 2048,
                "initial_temperature" : 5.,
                "reach_zero_temperature_after_n_episodes" : 1000,
                "temperature_policy" : None #array of temperatures to be used for each episode. After using all of them, temperature is set to 0.
            }
        
        hyper_parameters.update(**kwargs)

        if hyper_parameters["temperature_policy"] is None:
            initial_value = hyper_parameters["initial_temperature"]
            num_iterations = hyper_parameters["reach_zero_temperature_after_n_episodes"]
            exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
            hyper_parameters["temperature_policy"] = [initial_value * (exp_decay ** i) for i in range(num_iterations)]        

        self.hyper_parameters = hyper_parameters

        self.save_hyperparameters()

        #---Initialize environment---#
        self.env = env

        if isinstance(env, str):
            self.env = gym.make(self.env)

        if preprocess_env is not None:
            self.env = preprocess_env(self.env)
        
        self.env = wrap_env(self.env, video_callable=lambda episode_id: episode_id % save_video_every_n_episodes == 0)
        state_space_dim  = self.env.observation_space.shape 
        action_space_dim = self.env.action_space.n

        #---Initialize policy networks---#
        self.policy_net = Network(state_space_dim, action_space_dim)
        self.target_net = Network(state_space_dim, action_space_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict()) #Copy weights from policy network to the target network

        self.loss_fn = getattr(nn, hyper_parameters["loss_function"])()

        self.memory = ReplayMemory(capacity=hyper_parameters["replay_memory_capacity"])
        self.agent = Agent(self.env, self.memory, add_to_reward=add_to_reward, chosen_policy=chosen_policy)

        #---Populate memory---#
        for i in trange(hyper_parameters["min_samples_for_training"]):
            self.agent.play_step(self.policy_net)

        self.exploration_profile = hyper_parameters["temperature_policy"]

        self.total_reward = 0
        self.episode_reward = 0
        self.n_episodes = 0

        self.episode_history = []
        self.temp_history = []

        self.count_steps = count_steps
        self.update_temp_every_frame = update_temp_every_frame
        self.update_target_every_frame = update_target_every_frame

    def forward(self, x : "torch.Tensor") -> "torch.Tensor":
        out = self.policy_net(x)

        return out

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def training_step(self,
                      batch : Tuple["torch.Tensor", "torch.Tensor"],
                      idx_batch : int) -> "torch.Tensor":
        
        #---Generate a new sample---#

        #1. Set temperature
        temp_idx = self.n_episodes
        if self.update_temp_every_frame:
            temp_idx = self.global_step

        if self.n_episodes < len(self.exploration_profile):
            temperature = self.exploration_profile[temp_idx]
        else:
            temperature = 0.

        self.agent.set_temperature(temperature)

        #2. Play a step
        reward, done = self.agent.play_step(self.policy_net, device=self.get_device(batch))

        if self.count_steps:
            self.episode_reward += 1 #1 per step
        else:
            self.episode_reward += reward

        #3. Log if an episode has ended
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

            self.episode_history.append(self.total_reward)
            self.temp_history.append(temperature)
            
            if (not self.update_target_every_frame) and self.n_episodes % self.hyper_parameters["target_net_update_steps"] == 0:
                logging.info("Updating target network...")
                self.target_net.load_state_dict(self.policy_net.state_dict())
                #Update parameters of target network

            logging.info(f"EP: {self.n_episodes + 1} - SCORE: {self.total_reward} (Avg(10): {np.mean(self.episode_history[-10:]):.2f} - Avg(100): {np.mean(self.episode_history[-100:]):.2f}) - Temp: {temperature:.3f}")
            self.n_episodes += 1


        if self.update_target_every_frame and self.global_step % self.hyper_parameters["target_net_update_steps"] == 0:
            logging.info("Updating target network...")
            self.target_net.load_state_dict(self.policy_net.state_dict())
            #Update parameters of target network
        
        #4. Log general statistics
        self.log('total_reward', self.total_reward, prog_bar=True)
        self.log('global_step', self.global_step, prog_bar=True)
        self.log('n_episodes', self.n_episodes, prog_bar=True)
        self.log('temp', temperature, prog_bar=True)

        #---Compute loss---#
        #Goal: minimize distance between:
        #(i) reward + gamma * max_a (Q'(next_state, a))
        #(ii) Q(state, action)
        # where Q' is the "target_net" and Q the "policy_net"

        #1. Q-Values of the policy network
        states, actions, next_states, rewards, dones = batch

        q_values = self.policy_net(states) #Shape: (N, F)
        #N = batch_size; F = number of available actions

        #Select, between the F actions, the Q-value of the action that was actually performed [term (i)]
        state_action_values = q_values.gather(1, actions.unsqueeze(-1)) #Shape: (N, 1)

        #batch.action.unsqueeze(-1) converts batch.action to a "column" tensor ([[a], [b]...]), of shape (N, 1)
        #q_values.gather(1, actions)[i, j] => q_values[i, actions[i][j]]
        #Basically, for each row of q_values, take the entry with index given by the action specified for that row

        #2. Q-Values of the target network
        # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )

        with torch.no_grad():
            #self.target_net.eval()
            next_state_max_q_values = self.target_net(next_states).max(1)[0] #[0] selects only the values (torch.max returns also the indices)
            next_state_max_q_values[dones] = 0. #Zero the q-value for final states
    
        #---Expected Q-Values---#
        expected_state_action_values = rewards + (next_state_max_q_values * self.hyper_parameters["gamma"]) #term (ii)
        expected_state_action_values = expected_state_action_values.unsqueeze(1) #to "column" tensor

        #print(state_action_values.shape, expected_state_action_values.shape)

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        return loss
    
    def configure_optimizers(self):

        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.hyper_parameters["learning_rate"])
        
        return optimizer

    def train_dataloader(self):
        """
        Return samples for training from the replay memory.
        """

        batch_size = self.hyper_parameters["batch_size"]
        epoch_size = self.hyper_parameters["steps_per_epoch"] #How many steps in an epoch

        dataset = MemoryDataset(buffer=self.memory, epoch_size=epoch_size)
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None,
            pin_memory=True
        )

        return dataloader
