#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This is the coursework 2 for the Reinforcement Leaning course 2021 taught at Imperial College London (https://www.imperial.ac.uk/computing/current-students/courses/70028/)
# The code is based on the OpenAI Gym original (https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) and modified by Filippo Valdettaro and Prof. Aldo Faisal for the purposes of the course.
# There may be differences to the reference implementation in OpenAI gym and other solutions floating on the internet, but this is the defeinitive implementation for the course.

# Instaling in Google Colab the libraries used for the coursework
# You do NOT need to understand it to work on this coursework

# !pip install gym

from IPython.display import clear_output
clear_output()


# In[2]:


# Importing the libraries

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder    #records videos of episodes
import numpy as np
import matplotlib.pyplot as plt # Graphical library

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # get RTX3080 Ready

from collections import namedtuple, deque
from itertools import count
import math
import random
from gym.wrappers.frame_stack import FrameStack

clear_output()


# In[3]:


# Test cell: check ai gym  environment + recording working as intended

env = gym.make("CartPole-v1")
file_path = './video_test.mp4'
recorder = VideoRecorder(env, file_path)

observation = env.reset()
terminal = False
while not terminal:
  recorder.capture_frame()
  action = int(observation[2]>0)
  observation, reward, terminal, info = env.step(action)
  # Observation is position, velocity, angle, angular velocity

recorder.close()
env.close()


# In[4]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# defining replay buffer
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[5]:


class DQN(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputs, hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden-1)])
        self.output_layer = nn.Linear(hidden_size, outputs)
    
    def forward(self, x):
        x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)


# In[6]:



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # setting the imput of mini-batches    
    transitions = memory.sample(BATCH_SIZE) 

    batch = Transition(*zip(*transitions)) # seperate state action reward

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool) # transform the array of states into tensor so that GPU can handle
    

    # Can safely omit the condition below to check that not all states in the
    # sampled batch are terminal whenever the batch size is reasonable and
    # there is virtually no chance that all states in the sampled batch are 
    # terminal
    if sum(non_final_mask) > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
    else:
        non_final_next_states = torch.empty(0,state_dim).to(device)

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad(): #there will be a backward() called later
        # Once again can omit the conditional if batch size is large enough
        if sum(non_final_mask) > 0:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        else:
            next_state_values = torch.zeros_like(next_state_values)


    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = ((state_action_values - expected_state_action_values.unsqueeze(1))**2).sum()
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # Limit magnitude of gradient for update step
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()


# In[7]:


# Training
#setting parameters
NUM_EPISODES = 200
BATCH_SIZE = 256
GAMMA = 0.99
MEMORY_CAPACITY = 5000
INTEGRATED = 4 # number of previous states
# threshold_duration = 1000 this is not needed, the duration never exceed 500


epsilon = 0.09 # for epsilon greedy policy 
num_hidden_layers = 5
size_hidden_layers = 200
target_update_frequency = 10


# Get number of states and actions from gym action space
env = gym.make("CartPole-v1")
env = FrameStack(env, INTEGRATED)
env.reset()
state_dim = len(env.state) * INTEGRATED   #states are: x, x_dot, theta, theta_dot

n_actions = env.action_space.n
env.close()

policy_net = DQN(state_dim, n_actions, num_hidden_layers, size_hidden_layers).to(device)
target_net = DQN(state_dim, n_actions, num_hidden_layers, size_hidden_layers).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
                     
optimizer = optim.RMSprop(policy_net.parameters())

memory = ReplayBuffer(MEMORY_CAPACITY)


def select_action(state, current_eps=0):

    sample = random.random()
    eps_threshold = current_eps
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.

            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# In[8]:



file_path_2 = './video_train.mp4'
recorder = VideoRecorder(env, file_path_2)

observation = env.reset()
done = False

state = torch.tensor(env.state).float().unsqueeze(0)

duration_record = np.zeros(NUM_EPISODES)
reward_record = np.zeros(NUM_EPISODES)

#training
for i_episode in range(NUM_EPISODES):
    duration = 0 #reset duration 
    acc_r = 0
    if i_episode % target_update_frequency == 0:
        print("episode ", i_episode, "/", NUM_EPISODES)

    # Initialize the environment and state
    env.reset()
    state = torch.tensor(env.state).float().unsqueeze(0).to(device)
    state = state.repeat(1, INTEGRATED)

    for t in count():
        recorder.capture_frame()
        # Select and perform an action
        action = select_action(state, epsilon)

        states, reward, done, _ = env.step(action.item())
        frames_1d = np.array(states).flatten() 


        reward = torch.tensor([reward], device=device)
        duration += 1
        r = 1 
        acc_r = acc_r + r # to gain total reward

        # Observe new state
        if not done:
            next_state = torch.tensor(frames_1d).float().unsqueeze(0).to(device)
        else:
            next_state = None

        # Store the transition in memory    
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        # update target net
        if i_episode % target_update_frequency == 0:
           target_net.load_state_dict(policy_net.state_dict()) #transfer the data in policy net to target net
        if done:
            duration_record[i_episode] = duration
            reward_record[i_episode] = acc_r
            break
    recorder.close()
    env.close()
print("Episode duration: ", duration)   
        


print('Complete')

# duration analysis
# print("duration recorder shows: ", duration_record)
avg_duration = np.zeros(NUM_EPISODES)
acc_duration = 0
for i in range(NUM_EPISODES):
    acc_duration += duration_record[i]
    if i == 0:
        avg_duration[i] = 0
    else:     
        avg_duration[i] = acc_duration / i
avg_reward = np.zeros(NUM_EPISODES)
std_reward = np.zeros(NUM_EPISODES)
mix_n = np.zeros(NUM_EPISODES)
mix_p = np.zeros(NUM_EPISODES)
acc_reward = 0
for i in range(NUM_EPISODES):
    acc_reward += reward_record[i]
    std_reward[i] = np.std(reward_record[0:i])
    if i == 0:
        avg_reward[i] = 0
    else:     
        avg_reward[i] = acc_reward / i
        mix_p[i] = avg_reward[i] + std_reward[i]
        mix_n[i] = avg_reward[i] - std_reward[i]

# print("duration recorder shows: ", avg_duration)
plt.figure(1, figsize=(20,10))
plt.title('Durations and averange durations')
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.plot(duration_record,color='green', linestyle='-')
plt.plot(avg_duration,color='red', linestyle=':')

plt.figure(2, figsize=(20,10))
plt.title('reward and averange reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(reward_record,color='green', linestyle='-')
plt.plot(avg_reward,color='red', linestyle='--')
plt.plot(mix_p,color='blue', linestyle=':')
plt.plot(mix_n,color='blue', linestyle=':')

recorder.close()
env.close()
print("Episode duration: ", duration)

