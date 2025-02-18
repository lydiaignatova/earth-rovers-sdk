import time 
import requests
import base64
import pygame 
import threading
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import os
import json
from agentlace.action import ActionServer, ActionConfig, ActionClient
import utils 
import select
import numpy as np 

import pygame
import numpy as np
import time 


def calibrate_joystick():
    resting_values = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
    return resting_values

# Initialize Pygame
pygame.init()
pygame.joystick.init()

action_client = ActionClient(
    server_ip="localhost",
    config=utils.make_action_config("frodobot")
)

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
    
    linear, angular = -1 * axes_values[1], -0.6 * axes_values[0]
    return linear, angular


def send_joystick_commands():
    try:
        while True:
            time.sleep(0.2)
            linear, angular = get_joystick_input()
            if not(linear == 0.0 and angular == 0.0):
                print(f"GOT LINEAR {linear} ANGULAR {angular}")
                action_client.act("action_vw", np.array([linear, angular]))
    except KeyboardInterrupt:
        print("Exiting joystick control")

# Example usage:
if __name__ == "__main__":
    send_joystick_commands()

# Clean up when done
pygame.quit()
