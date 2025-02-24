# generic imports
import time
import numpy as np
from PIL import Image
import os
import sys
import io
from dataclasses import dataclass
from typing import Optional
import time 
import logging

import argparse

# data loading
from absl import app, flags, logging as absl_logging

import utils
import requests
import json
import pygame 
import pickle


MAX_TRAJ_LEN = 100 # 3 times a second * 30 seconds = 90 long
STEPS_TRY = 60
GOAL_DIST = STEPS_TRY // 2 # 4 - 10 # normal around this with 5 std 

def calibrate_joystick():
    resting_values = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    return resting_values

# Initialize Pygame
pygame.init()
pygame.joystick.init()

# Check if any joysticks are available
if pygame.joystick.get_count() < 1:
    print("No joystick connected")
    exit() 
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Joystick detected: {joystick.get_name()}")

calibrated_axes = calibrate_joystick()

def get_joystick_input():

    pygame.event.pump()  # Make sure to process events
    
    num_axes = joystick.get_numaxes()  # Get the number of axes on the joystick
    axes_values = []
    
    # 4 axes when in D mode 
    for i in range(num_axes):
        axis_value = joystick.get_axis(i)
        axes_values.append(axis_value - calibrated_axes[i])
    
    linear, angular = -1 * axes_values[1], -1 * axes_values[0]
    return linear, angular


class Recorder():
    def __init__(self, base_url, tick_rate):

        self.tick_rate = tick_rate

    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()    
        
        while True:
            cur_time = time.time()
            linear, angular = get_joystick_input()
            output = self.control_no_server(linear, angular)                      

    def control_no_server(self, linear, angular):
        data = json.dumps({"command": {"linear": linear, "angular": angular}})
        response = requests.post("http://127.0.0.1:8000/control", data=data)
        response = response.json()
        if response["message"] == 'Command sent successfully':
            print(response["message"], "linear", linear, "angular", angular)
            return True 
        else:
            return False
            

if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING)
    absl_logging.set_verbosity("WARNING")

    parser = argparse.ArgumentParser(description='My Python Script')
    parser.add_argument('--base_url', type=str, default="localhost",  help='What IP to connect to a robot action server on')
    parser.add_argument('--tick_rate', type=int, default=3, help = "How frequently to send joystick commands")
    args = parser.parse_args()

    Recorder(base_url= args.base_url,
             tick_rate = args.tick_rate,
          ).run() 
