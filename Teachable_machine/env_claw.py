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

class environment():
    def __init__(self):
        self.cmd_on = os.getcwd() + '\claw_machine\ClawMachineSimulator.exe'
        self.cmd_off = 'taskkill /f /im ClawMachineSimulator.exe'
        self.move = [0, 0]

    def reset(self): #純截圖
        flatten = nn.Flatten(0, 1)
        image = pyautogui.screenshot()
        cv_screen = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        cv_screen = cv2.resize(cv_screen,(192,108))
        img_gray_tensor = torch.tensor(cv_screen)
        flat_image_gray = flatten(img_gray_tensor)
        to_one = 1/255
        simple_img = torch.mul(flat_image_gray, to_one)
        return image

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
            

            flatten = nn.Flatten(0, 1)
            image = pyautogui.screenshot()
            cv_for_reward = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            reward, end = claw_reward.reward_calculate(False, count, cv_for_reward)

            cv_screen = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            cv_screen = cv2.resize(cv_screen,(192,108))
            img_gray_tensor = torch.tensor(cv_screen)
            flat_image_gray = flatten(img_gray_tensor)
            to_one = 1/255
            simple_img = torch.mul(flat_image_gray, to_one)
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

            time.sleep(14-shot_time) #這裡是夾完的時刻 要決定要多久後拍照
            flatten = nn.Flatten(0, 1)
            image2 = pyautogui.screenshot()
            cv_screen = cv2.cvtColor(np.asarray(image2), cv2.COLOR_RGB2GRAY)
            cv_screen = cv2.resize(cv_screen,(192,108))
            img_gray_tensor = torch.tensor(cv_screen)
            flat_image_gray = flatten(img_gray_tensor) #應該是next_state
            to_one = 1/255
            simple_img = torch.mul(flat_image_gray, to_one)

        #next_state, reward, done, info 格式應該是tensor, 數字, 數字, 沒差
        return simple_img, reward, done, 1

     


