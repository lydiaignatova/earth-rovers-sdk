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

import select



class JoyStickInterface: 
    pass

class APIInterface:
    def __init__(self, 
                 api_key, 
                 base_url, 
                 command_interface, 
                 joystick_id=0, 
                 frame_rate=30, 
                 save_dir= None,
                 data_rate=None):
        if save_dir is not None:
            self.save_dir = save_dir
            self.save_imgs = True
        else:
            self.save_imgs = False
        self.base_url = base_url

        # Sending commands 
        self.command_interface = command_interface
        self.joystick_id = joystick_id

        self.vel_lin = 0.0
        self.vel_ang = 0.0

        if self.command_interface == "joystick":
            pygame.init()
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(self.joystick_id)
            self.joystick.init()
        elif self.command_interface == "keyboard":
            pygame.init()
            # raise NotImplementedError("Keyboard interface not implemented yet")
            
        # Receive data
        self.frame_rate = frame_rate
        self.data_rate = data_rate

        # Start the API interface
        self.start()
    
    def decode_from_base64(self, base64_string):
        image = Image.open(BytesIO(base64.b64decode(base64_string)))
        return image

    def send_velocity_command(self, linear, angular):
        print("Linear: ", linear, "Angular: ", angular)
        url = f"{self.base_url}/control"
        data = json.dumps({"command": {"linear": linear, "angular": angular}})
        response = requests.post(url, data=data)
        response = response.json()
        if response["message"] == 'Command sent successfully':
            return True 
        else:
            return False
    
    def data_request(self, lock):
        url = f"{self.base_url}/data"
        response = requests.get(url)
        response = response.text
        with lock:
            self.current_data = response
        return True
    
    def image_request(self, lock):
        url = f"{self.base_url}/screenshot"
        response = requests.get(url) 
        response = response.json()
        front_frame = self.decode_from_base64(response["front_frame"])
        rear_frame = self.decode_from_base64(response["rear_frame"])
        map_frame = self.decode_from_base64(response["map_frame"])
        timestamp = response["timestamp"]
        with lock:
            self.current_front_frame = (front_frame, timestamp)
            self.current_rear_frame = (rear_frame, timestamp) 
            self.current_map = (map_frame, timestamp)
        # save the image if saving
        if self.save_imgs:
            self.current_front_frame[0].save(f"{self.save_dir}/{timestamp}_front.jpg")

        return True
    
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

    def send_key_commands(self):
        print("Enter w a s d commands, with q to quit.")
        i, o, e = select.select([sys.stdin], [], [], 0.001)
    
        if i:  # If there's input within the timeout
            command = sys.stdin.readline().strip().lower()

            if command == 'w':  # Move forward
                self.vel_lin = min(self.vel_lin + 0.1, 1.0)
            elif command == 's':  # Move backward
                self.vel_lin = max(self.vel_lin - 0.1, 0.0)
            elif command == 'a':  # Turn left
                self.vel_ang = min(self.vel_ang + 0.1, 1.0)
            elif command == 'd':  # Turn right
                self.vel_ang = max(self.vel_ang - 0.1, 0.0)
            elif command == 'q':  # Quit
                print("Quitting command input.")
                self.vel_lin = self.vel_ang = 0.0
            else:
                print("Invalid command. Please use 'w', 'a', 's', 'd', or 'q'.")
                self.vel_lin = self.vel_ang = 0.0
        else:
            print("No input received, sending current velocity...")

        # Continuously send the current velocity values even if no new input is received
        self.send_velocity_command(self.vel_lin, self.vel_ang)
        # print(f"Current velocity: linear {self.vel_lin}, angular {self.vel_ang}")

    
        # if command == 'w':  # Move forward
        #     self.vel_lin = min(self.vel_lin + 0.1, 1.0) # Set your desired speed
        # elif command == 's':  # Move backward
        #     self.vel_lin = max(self.vel_lin - 0.1, 0.0)
        # elif command == 'a':  # Turn left
        #     self.vel_ang = min(self.vel_ang + 0.1, 1.0)
        # elif command == 'd':  # Turn right
        #     self.vel_ang = max(self.vel_ang - 0.1, 0.0)
        
        # elif command == 'q':  # Quit
        #     print("Quitting command input.")
        #     self.vel_lin = self.vel_ang = 0.0
        # else:
        #     print("Invalid command. Please use 'w', 'a', 's', 'd', or 'q'.")
        #     linear = self.vel_ang = 0.0

        # print(f"Current velocity {self.vel_lin} linear {self.vel_ang} angular")

        # self.send_velocity_command(self.vel_lin, self.vel_ang)

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
    api_interface = APIInterface(api_key, "http://localhost:8000", 
                                 "keyboard", 
                                 frame_rate=30, 
                                #  save_dir = "front",
                                 data_rate=30)
    api_interface.start()




    
    
