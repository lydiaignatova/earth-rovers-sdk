## HOW TO USE THE SDK! 

import time 
import requests
import base64
# import pygame 
import threading
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import os
import json
import tensorflow as tf
from agentlace.action import ActionServer, ActionConfig

## WORKING W SDK 
def data_request(url):
    url = f"{url}/data"
    response = requests.get(url)
    response = response.text
    return response


def observation_format(observation_key_type):
    if observation_key_type == "frodobot":
        return    {
            "front_frame": tf.TensorSpec((), tf.string, name="front_frame"),
            "rear_frame": tf.TensorSpec((), tf.string, name="rear_frame"),
            "map_frame": tf.TensorSpec((), tf.string, name="map_frame"),

            "timestamp_img": tf.TensorSpec((), tf.float64, name="timestamp_img"),
            "timestamp_data": tf.TensorSpec((), tf.float64, name="timestamp_data"),

            "battery": tf.TensorSpec((), tf.int32, name="battery"),
            "signal_level": tf.TensorSpec((), tf.int32, name="signal_level"),
            "orientation": tf.TensorSpec((), tf.int32, name="orientation"),
            "lamp": tf.TensorSpec((), tf.int32, name="lamp"),
            "speed": tf.TensorSpec((), tf.float32, name="speed"),
            "gps_signal": tf.TensorSpec((), tf.float32, name="gps_signal"),
            "latitude": tf.TensorSpec((), tf.float64, name="latitude"),  # for precision with coordinates
            "longitude": tf.TensorSpec((), tf.float64, name="longitude"), # for precision with coordinates
            "vibration": tf.TensorSpec((), tf.float32, name="vibration"),
            # "timestamp": tf.TensorSpec((), tf.float64, name="timestamp"), # 64-bit for timestamp precision
            "accels": tf.TensorSpec((6, 4), tf.float64, name="accels"),   # 6x4 matrix for accelerometer data
            "gyros": tf.TensorSpec((5, 4), tf.float64, name="gyros"),     # 5x4 matrix for gyroscope data
            "mags": tf.TensorSpec((1, 4), tf.float64, name="mags"),       # 1x4 matrix for magnetometer data
            "rpms": tf.TensorSpec((5, 5), tf.float64, name="rpms"),       # 5x5 matrix for RPM data

            "action_state_source": tf.TensorSpec((), tf.string, name="action_state_source"),
            "last_action_linear": tf.TensorSpec((3,), tf.float32, name="last_action_linear"),
            "last_action_angular": tf.TensorSpec((3,), tf.float32, name="last_action_angular"),
        }
    else: 
        raise ValueError(f"Unknown observation config type {observation_key_type}")

def make_action_config(action_config_type):
    if action_config_type == "frodobot":
        return ActionConfig(
            port_number=1111,
            action_keys=["action_vw"],
            observation_keys=list(observation_format(action_config_type).keys()),
        )
    else:
        raise ValueError(f"Unknown action config type {action_config_type}")


    
def image_request(url, decode = True):
    url = f"{url}/screenshot"
    response = requests.get(url) 
    response = response.json()

    if decode:
        front_frame = decode_from_base64(response["front_frame"])
        rear_frame = decode_from_base64(response["rear_frame"])
        map_frame = decode_from_base64(response["map_frame"])
    else:
        front_frame = response["front_frame"]
        rear_frame = response["rear_frame"]
        map_frame = response["map_frame"]
    timestamp = response["timestamp"]

    return {
        "front": front_frame,
        "rear": rear_frame,
        "map": map_frame,
        "timestamp": timestamp
    }

def send_velocity_command(url, linear, angular):
    url = f"{url}/control"

    data = json.dumps({"command": {"linear": linear, "angular": angular}})
    response = requests.post(url, data=data)
    response = response.json()
    if response["message"] == 'Command sent successfully':
        return True 
    else:
        return False


## PROCESSING DATA
def decode_from_base64(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    return image


    def send_joy_commands(self):
        events = pygame.event.get()
        
        if len(events) > 0:
            event = events[0]
            print(f"got event {event}")
            print(f"event type {type(event)} with event type {event.type}")
        else:
            return False
        

        if event.type == pygame.JOYAXISMOTION:
            linear = self.joystick.get_axis(4)*-1
            angular = self.joystick.get_axis(3)*-1
            self.send_velocity_command(linear, angular)
        elif event.type == 1538: # lower left hand size joystick thing
            print(f"value is {event.value} with type {type(event.value)}")
            linear =  0.2 * event.value[1]
            angular = -0.4 * event.value[0]
            self.send_velocity_command(linear, angular)

        return True

    def image_loop(self, lock):
        while True:
            self.image_request(lock)
            time.sleep(1/self.frame_rate)
    
    def data_loop(self, lock):
        while True:
            self.data_request(lock)
            time.sleep(1/self.data_rate)

    # Main loop which receives data and sends commands
    def start(self):
        print("Starting API Interface")
        lock = threading.Lock()
        image_thread = threading.Thread(target=self.image_loop, args=(lock,))
        data_thread = threading.Thread(target=self.data_loop, args=(lock,))
        image_thread.start()
        data_thread.start()
        try: 
            while True:
                if self.command_interface == "joystick":
                    self.send_joy_commands()
                elif self.command_interface == "keyboard":
                    self.send_key_commands()
        except KeyboardInterrupt:
            image_thread.join()
            data_thread.join()
            pygame.quit()
            print("Exiting API Interface")
    
if __name__ == "__main__":
    api_key = os.getenv("SDK_API_TOKEN")
    api_interface = APIInterface(api_key, "http://localhost:8000", "joystick", frame_rate=30, data_rate=30)
    api_interface.start()




    
    
