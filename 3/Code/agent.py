import torch
import torch.nn as nn

from data import ReplayMemory

import numpy as np
import gym
from typing import Callable, Tuple
import random
import logging

class Agent:

    def __init__(self,
                 env : "gym.Env",
                 memory : "ReplayMemory",
                 chosen_policy : str = "softmax",
                 add_to_reward : Callable[["np.ndarray", float], float] = None
                 ) -> None:
        """
        Represents an Agent acting in an environment `env`, and storing
        replays in a `memory` buffer.

        Parameters
        ----------
        env : "gym.Env"
            Environment the agent acts in
        memory : "ReplayMemory"
            Buffer to store replays,
            i.e. tuples (state, action, next_state, reward, done)
        add_to_reward : function(state, reward) -> new_reward
            A custom function that is added to the reward received from the environment. 
            It can be used to fine-tune the training, allowing a faster convergence.
        """

        self.env = env
        self.memory = memory
        self.state = self.env.reset()
        self.add_to_reward = add_to_reward

        self.temperature = 1. #Used as temperature by "softmax" policy, or as
        #epsilon by "epsilon-greedy".

        self.chosen_policy = chosen_policy


    @torch.no_grad()
    def play_step(self,
                  net : "nn.Module",
                  device : "torch.device" = 'cpu') -> Tuple[float, bool]:
        """
        Uses `net` as a policy network to decide the action to be performed. 
        That action is carried on, and a memory of the state transition 
        is stored in the replay buffer.

        Parameters
        ----------
        net : "nn.Module"
            Policy network. `net(state)` should return a tensor of shape
            (action_space_dim,) containing the Q-value for each available action.
        device : "torch.device"
            Device used for training

        Returns
        -------
        reward : float
            Reward obtained from performing the chosen action.
        done : bool
            True if a final state has been reached, and no further action is allowed.    
        """
        
        #---Compute Q-values using the policy network---#
        state_tensor = torch.tensor(self.state, dtype=torch.float32, device=device)
        net_out = net(state_tensor.unsqueeze(0)) #unsqueeze(0)

        #---Obtain an action using the policy---#
        action = self.policy(net_out)

        #---Perform the action and observe the results---#
        next_state, reward, done, info = self.env.step(action)

        if self.add_to_reward is not None:
            reward += self.add_to_reward(self.env.unwrapped.state, reward)
        
        #---Add to memory---#
        self.memory.push(self.state, action, next_state, reward, done)

        #---Update state---#
        self.state = next_state

        if done:
            self.state = self.env.reset()

        return reward, done

    def set_temperature(self, temperature):
        """
        Set the temperature for the softmax policy.
        """

        self.temperature = temperature

    @torch.no_grad()
    def policy(self, q_values : "torch.tensor"):
        """
        Softmax policy for choosing an action based on the `q_values`.
        """

        if self.chosen_policy == "softmax":
            if self.temperature == 0: #If temperature is 0, pick the best action
                return int(q_values.argmax())
            
            temperature = max(self.temperature, 1e-8) # set a minimum to the temperature for numerical stability
            softmax_out = nn.functional.softmax(q_values.flatten() / temperature, dim=0).cpu().numpy()
              
            # Sample the action using softmax output as mass pdf
            all_possible_actions = np.arange(0, len(softmax_out))
            
            action = np.random.choice(all_possible_actions, p=softmax_out) 
            # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
        elif self.chosen_policy == "epsilon-greedy":
            if self.temperature >= 1:
                self.temperature = 1

            best_action = int(q_values.argmax())

            if self.temperature <= 0:
                return best_action

            n_actions = len(q_values.flatten())
            if random.random() < self.temperature: 
                action = random.choice([a for a in range(n_actions) if a != best_action])
            else:
                action = best_action

        else:
            raise NotImplementedError("This policy is not implemented")

        return action

def test_agent():
    env = gym.make("CartPole-v1")

    net = lambda state : torch.rand(env.action_space.n) # Simulate a network returning random q-values
    memory = ReplayMemory(capacity = 500)

    add_to_reward = lambda state, reward : -np.abs(state[0])

    agent = Agent(env, memory, add_to_reward=add_to_reward)

    done = False
    n_frames = 0
    
    while not done:
        reward, done = agent.play_step(net)
        n_frames += 1

    assert n_frames >= 1
    assert len(memory) == n_frames
    

