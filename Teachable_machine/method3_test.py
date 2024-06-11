import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
from tqdm import tqdm
from env_claw import environment
import choose_action
import random
import pyautogui
total_rewards = []

def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    rewards = []
    epsilon = 0.5
    machine_reset_period = 12
    for epi in range(10):
        if epi % machine_reset_period == 0:
            if epi != 0:
                env.close()
            env.machine_reset()
        state = env.reset()
        count = 0
        while True:
            # env.render() #Uncomment for visualizing the process
            count += 1

            if np.random.rand() < (1-epsilon):
                action = random.choice([0,0,0,0,1,2,3,3,3,3,4])
            else:
                image = pyautogui.screenshot()
                action = choose_action.choose(image)
            next_state, reward, done, _ = env.step(action, count)
            if done:
                rewards.append(reward/10-count) #還要加
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")

if __name__ == "__main__":
    env = environment()     

    # testing section:
    test(env)
    env.close()
