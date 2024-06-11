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
from env_claw import environment
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
    def __init__(self,  num_actions, hidden_layer_size=10371):
        super(Net, self).__init__()
        self.input_state = 20736  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 10371)  # input layer
        self.fc2 = nn.Linear(10371, hidden_layer_size)  # hidden layer
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

        b_s, b_a, b_r, b_ns, b_d = self.buffer.sample(self.batch_size)
        #b_s = torch.tensor(b_s, dtype=torch.float)
        b_s_2d = torch.cat((b_s[0].reshape(1, -1), b_s[1].reshape(1, -1)), 0)
        for i in range(2, len(b_s)):
            b_s_2d = torch.cat((b_s_2d, b_s[i].reshape(1, -1)), 0)
        b_a = torch.tensor(np.array(b_a).reshape(self.batch_size, 1), dtype=torch.long)
        b_r = torch.tensor(np.array(b_r).reshape(self.batch_size, 1), dtype=torch.float)
        #b_ns = torch.tensor(b_ns, dtype=torch.float)
        b_ns_2d = torch.cat((b_ns[0].reshape(1, -1), b_ns[1].reshape(1, -1)), 0)
        for i in range(2, len(b_ns)):
            b_ns_2d = torch.cat((b_ns_2d, b_ns[i].reshape(1, -1)), 0)
        b_a = torch.tensor(np.array(b_a).reshape(self.batch_size, 1), dtype=torch.long)
        b_d = torch.tensor(np.array(b_d).reshape(self.batch_size, 1), dtype=torch.bool)

        q_values = self.evaluate_net(b_s_2d).gather(1, b_a)
        max_next_q_value = self.target_net(b_ns_2d).detach()
        q_targets = b_r + self.gamma * max_next_q_value.max(1)[0].view(self.batch_size, 1) * (~b_d).reshape(self.batch_size, 1)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % 500 == 0:
            torch.save(self.target_net.state_dict(), "./Tables/DQN_"+str(self.count)+".pt")
            #torch.save(self.evaluate_net.state_dict(), "./Tables/DQNe_"+str(self.count)+".pt")

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
            #a = np.arange(5)
            if np.random.rand() < (1-self.epsilon):
                r = np.random.rand()
                if r >= 0 and r < 0.06:
                    action = 1 
                elif r < 0.12:
                    action = 2
                elif r < 0.18:
                    action = 4
                elif r < 0.59:
                    action = 0
                elif r <= 1:
                    action = 3
                #action = np.random.choice(a)
            else:
                #state = torch.tensor(np.array([state]), dtype=torch.float)
                state_q = self.evaluate_net(state)
                action = np.argmax(state_q).item()

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

        state = self.env.reset()
        state = torch.tensor(np.array([state]), dtype=torch.float)
        state_q = self.target_net(state)
        return torch.max(state_q)


def train(env, continue_run):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    agent = Agent(env)
    episode = 1000
    machine_reset_period = 24
    rewards = []
    if continue_run == 1:
        agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
        agent.evaluate_net,load_state_dict(torch.load("./Tables/DQNe.pt"))
        with open('agent_count.txt', 'r') as f:
            c = f.readline()
            agent.count = int(c)

    for epi in range(episode):
        if epi % machine_reset_period == 0:
            if epi != 0:
                env.close()
            env.machine_reset()
        state = env.reset()
        count = 0
        save = 0
        if len(agent.buffer) >= 1000 or continue_run == 0:
            save = 1
        while True:
            count += 1
            if save == 1:
                agent.count += 1
                
            if agent.count < 2000:
                agent.epsilon = 0.02
            elif agent.count < 4000:
                agent.epsilon = 0.5
            elif agent.count < 7000:
                agent.epsilon = 0.8
            else:
                agent.epsilon = 0.95

            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action, count)
            agent.buffer.insert(state, int(action), reward, next_state, int(done))
            
            if len(agent.buffer) >= 1000:
                agent.learn()

            with open('agent_count.txt', 'w') as f: 
                f.write(str(agent.count))

            with open('rewards.txt', 'a') as f: 
                    f.write(str(round(reward, 3))+' ') #還要加rewards
            if done:
                with open('rewards.txt', 'a') as f: 
                    f.write('\n'+str(reward/10-count)+'\n') #還要加rewards
            if done and save:
                break
            
            state = next_state
    total_rewards.append(rewards)

if __name__ == "__main__":
    env = environment()  
    os.makedirs("./Tables", exist_ok=True)

    # training section:
    c = input("Continue(1) or not(0): ")
    c = int(c)
    train(env, c)        
    env.close()

    #os.makedirs("./Rewards", exist_ok=True)
    #np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))