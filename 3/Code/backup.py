#Baseline (random agent)----------------------
#---Test of rendering---#
#CartPole environment, with random actions at every turn

# Initialize the Gym environment
env = gym.make('CartPole-v1') 
env.seed(0) # Set a random seed for the environment (reproducible results)

# This is for creating the output video in Colab, not required outside Colab
env = wrap_env(env, video_callable=lambda episode_id: True)

# Let's try for a total of 10 episodes
for num_episode in range(10): 
    # Reset the environment and get the initial state
    state = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    done = False
    # Go on until the pole falls off or the score reach 490
    while not done and score < 490:
      # Choose a random action
      action = random.choice([0, 1])
      # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
      next_state, reward, done, info = env.step(action)
      # Visually render the environment (optional, comment this line to speed up the simulation)
      env.render()
      # Update the final score (+1 for each step)
      score += reward 
      # Set the current state for the next iteration
      state = next_state
      # Check if the episode ended (the pole fell down)
    # Print the final score
    print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}") 

env.close()

#-------------------------------------------------#

### Define exploration profile
initial_value = 5
num_iterations = 1000
exp_decay = np.exp(-np.log(initial_value) / num_iterations * 6) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

### Plot exploration profile
plt.figure(figsize=(5,5))
plt.plot(exploration_profile)
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Exploration profile (Softmax temperature)')

#-------------------------------------------------#

pos_weight = 1
reward - pos_weight * np.abs(self.state[0]) 

    #TODO Maybe add a "on_epoch_end" callback to policies
    # @torch.no_grad()
    # def policy(self, q_values, epsilon):
        
    #     best_action = int(q_values.argmax())

    #     if random.random() < epsilon: #can't we use a torch function here?
    #         action = random.choice([a for a in range(len(q_values)) if a != best_action])
    #     else:
    #         action = best_action

    #     return action

#--------------------------------------------------#
#Loss

#Example of computing the loss
dataset = MemoryDataset(buffer=agent.memory, epoch_size=4)
dataloader = DataLoader(dataset=dataset, sampler=None, batch_size=4)

batch = next(iter(dataloader))
states, actions, next_states, rewards, dones = batch

policy_net = DQN(4, 2)
target_net = DQN(4, 2)

gamma = .97

policy_net.eval() #Just for this example
target_net.eval() 

loss_fn = nn.SmoothL1Loss()

with torch.no_grad():
    q_values = policy_net(states) #Shape: (N, F)
    #N = batch_size; F = number of available actions

    #Select, between the F actions, the Q-value of the action that was actually performed
    state_action_values = q_values.gather(1, actions.unsqueeze(-1)) #Shape: (N, 1)

    #batch.action.unsqueeze(-1) converts batch.action to a "column" tensor ([[a], [b]...]), of shape (N, 1)
    #q_values.gather(1, actions)[i, j] => q_values[i, actions[i][j]]
    #Basically, for each row of q_values, take the entry with index given by the action specified for that row

    q_values_target = target_net(next_states)
    next_state_max_q_values = q_values_target.max(dim=1)[0] #[0] selects only the values (torch.max returns also the indices)
    next_state_max_q_values[dones != False] = 0. #Zero the q-value for final states

    #---Expected Q-Values---#
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1) #to "column" tensor

print(loss_fn(state_action_values, expected_state_action_values))

#So... a brief list of mistakes
#-When resetting the environment, save the new "first" state somewhere (e.g. self.state). Don't discard it for no reason
#-Updating the target net must be done every n episodes, not every n batches!
#-When computing the loss, you use next_state too.


#TODO:
#Re-organize code into separate .py files. Use one notebook for each of the 3 main tasks
#Add comments with explanations on what you are doing. Cite the example used for adapting RL to pytorch lightning (https://github.com/PyTorchLightning/pytorch-lightning/blob/bc2c2db2bfc4922691ed8d523fc655b9c0fa237c/pl_examples/domain_templates/reinforce_learn_Qnet.py#L229)
#Once the code is good, test it on another gym.env by re-using the pl environment. Make it reusable! Copy-pasting code is bad!

#After that, focus on the pixels. This will likely require some code to be adjusted, so not everything may be reusable. 

Some interesting papers & resources:
- "Rainbow": basically how to merge all improvements to DQN: https://arxiv.org/pdf/1710.02298.pdf

- "Prioritized experience replay" https://arxiv.org/pdf/1511.05952.pdf

- "Playing Atari with Deep Reinforcement Learning", the original paper by DeepMind https://arxiv.org/pdf/1312.5602v1.pdf

- PyTorch Tutorial for Cartpole with pixels: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

- Another tutorial with pixels: https://rubenfiszel.github.io/posts/rl4j/2016-08-24-Reinforcement-Learning-and-DQN.html

- Git on challenges for raw pixels: https://github.com/pytorch-rl/deepq
- https://pylessons.com/CartPole-PER-CNN/
