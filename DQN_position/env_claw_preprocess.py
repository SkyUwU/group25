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
import claw_reward
import claw_state

class environment():
    def __init__(self):
        self.cmd_on = os.getcwd() + '\claw_machine\ClawMachineSimulator.exe'
        self.cmd_off = 'taskkill /f /im ClawMachineSimulator.exe'
        self.move = [0, 0]

    def reset(self): 
        image = pyautogui.screenshot()
        cv_screen = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        claw_x, claw_y, return_x, return_y = claw_state.state_calculate(cv_screen)
        tensor_xy = torch.tensor([claw_x, claw_y, return_x, return_y])
        return tensor_xy

    def machine_reset(self): #重開夾娃娃機
        os.startfile(self.cmd_on)
        time.sleep(5)
        pyautogui.moveTo(1070, 770, duration = 0.2)
        pyautogui.click(clicks=1)
        time.sleep(11)
        pydirectinput.press('shiftleft')

        pyautogui.keyDown('w')
        time.sleep(5)
        pyautogui.keyUp('w')
        
        pyautogui.keyDown('space')
        time.sleep(4)
        pyautogui.keyUp('space')
        time.sleep(14)

    def close(self): #關掉夾娃娃機
        cl = os.system(self.cmd_off)
        
    def step(self, action, count):
        done = 0
        move_size = 0.2
        if action != 4:
            if action == 0:
                pyautogui.keyDown('w')
                time.sleep(move_size)
                pyautogui.keyUp('w')
            elif action == 1:
                pyautogui.keyDown('s')
                time.sleep(move_size)
                pyautogui.keyUp('s')
            elif action == 2:
                pyautogui.keyDown('a')
                time.sleep(move_size)
                pyautogui.keyUp('a')
            elif action == 3:
                pyautogui.keyDown('d')
                time.sleep(move_size)
                pyautogui.keyUp('d')
            
            image = pyautogui.screenshot()
            cv_for_reward = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            reward, end = claw_reward.reward_calculate(False, count, cv_for_reward)

            claw_x, claw_y, return_x, return_y = claw_state.state_calculate(cv_for_reward)
            tensor_xy = torch.tensor([claw_x, claw_y, return_x, return_y])
        else:
            done = 1
            end = 0
            shot_time = 0

            pyautogui.keyDown('space')
            time.sleep(4)
            pyautogui.keyUp('space')

            while end == 0:
                for_reward = pyautogui.screenshot()
                cv_for_reward = cv2.cvtColor(np.asarray(for_reward), cv2.COLOR_RGB2BGR)
                reward, end = claw_reward.reward_calculate(True, count, cv_for_reward)
                time.sleep(0.3)
                shot_time += 0.3
                
            if shot_time >= 14:
                shot_time = 13.9

            time.sleep(14-shot_time) #這裡是夾完的時刻 要決定要多久後拍照
            image = pyautogui.screenshot()
            cv_screen = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            claw_x, claw_y, return_x, return_y = claw_state.state_calculate(cv_screen)
            tensor_xy = torch.tensor([claw_x, claw_y, return_x, return_y])

        #next_state, reward, done, info 格式應該是tensor, 數字, 數字, 沒差
        return tensor_xy, reward, done, 1

     


