import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm
import time
import pyautogui
import pydirectinput
import cv2
from env_claw_preprocess import environment
total_rewards = []


class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''
    def __init__(self,  num_actions, hidden_layer_size=10):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.        
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 5  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method using to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None (Don't need to return anything)
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # Begin your code
        b_s, b_a, b_r, b_ns, b_d = self.buffer.sample(self.batch_size)
        #b_s = torch.tensor(np.array(b_s), dtype=torch.float)
        b_a = torch.tensor(np.array(b_a).reshape(self.batch_size, 1), dtype=torch.long)
        b_r = torch.tensor(np.array(b_r).reshape(self.batch_size, 1), dtype=torch.float)
        #b_ns = torch.tensor(np.array(b_ns), dtype=torch.float)
        b_d = torch.tensor(np.array(b_d).reshape(self.batch_size, 1), dtype=torch.bool)

        q_values = self.evaluate_net(b_s).gather(1, b_a)
        max_next_q_value = self.target_net(b_ns).detach()
        q_targets = b_r + self.gamma * max_next_q_value.max(1)[0].view(self.batch_size, 1) * (~b_d).reshape(self.batch_size, 1)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        # End your code
        if self.count % 100 == 0:
            torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        Parameters:
            self: the agent itself.
            state: the current state of the enviornment.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # Begin your code
            a = np.arange(5)
            if np.random.rand() < (1-self.epsilon):
                action = np.random.choice(a)
            else:
                #state = torch.tensor(np.array([state]), dtype=torch.float)
                state_q = self.evaluate_net(state)
                action = np.argmax(state_q).item()
            # End your code
        return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state        
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        state = self.env.reset()
        #state = torch.tensor(np.array([state]), dtype=torch.float)
        state_q = self.target_net(state)
        return torch.max(state_q)
        # End your code

def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    machine_reset_period = 12
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables_prep/DQN.pt"))
    for epi in range(100):
        if epi % machine_reset_period == 0:
            if epi != 0:
                env.close()
            env.machine_reset()
        state = env.reset()
        count = 0
        while True:
            # env.render() #Uncomment for visualizing the process
            count += 1
            Q = testing_agent.target_net.forward(
                torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            next_state, reward, done, _ = env.step(action, count)
            if done:
                rewards.append(reward/10-count) #還要加
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    env = environment()     

    # testing section:
    test(env)
    env.close()
