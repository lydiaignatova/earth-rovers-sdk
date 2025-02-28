import threading
import time
import requests

import numpy as np 
from collections import deque

import os
import utm 
import math 
import base64
from io import BytesIO
from PIL import Image

from openai import OpenAI

from google import genai


import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("Agg")  # Use a non-interactive backend to work with multithreading

def convert_angle_compass_to_cartesian(compass):
    return(((360 - compass) % 360) + 90) % 360


def calculate_relative_position(x_a, y_a, x_b, y_b):
    delta_x = x_b - x_a
    delta_y = y_b - y_a
    return delta_x, delta_y

def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
    # Apply the rotation matrix for the local frame
    relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
    relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
    
    return relative_x, relative_y

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        img_b64_bytes = base64.b64encode(image_file.read())
        img_b64_str = img_b64_bytes.decode("utf-8")
    return img_b64_str

def resize_image_byte(img_b64_str): 
    img = byte_to_pil(img_b64_str)

    new_size = (img.width // 2, img.height // 2)
    resized_img = img.resize(new_size)
    
    return pil_to_byte(resized_img)


def resize_images_byte(byte_imgs):
    return [resize_image_byte(img_b64_str) for img_b64_str in byte_imgs]
    

def byte_to_pil(byte_img):
    img_data = base64.b64decode(byte_img)
    img = Image.open(BytesIO(img_data))
    return img 


def pil_to_byte(pil_img):
    pil_img = pil_img.convert('RGB') # no more transparency 
    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")  
    buffer.seek(0)
    
    base64_img = base64.b64encode(buffer.read()).decode('utf-8')
    return base64_img


def overlay_imgs(pil_img1, pil_img2):
    pil_img1 = pil_img1.convert("RGBA")
    pil_img2 = pil_img2.convert("RGBA")
    """ Put second pic on first """
    pil_img2 = pil_img2.resize(pil_img1.size)
    result = Image.alpha_composite(pil_img1, pil_img2)
    return result 


class LLMHelper():

    def __init__(self, map_bounds, goal_pose, obs_rate, help_rate, api_name = "GPT"):
        if api_name == "GPT":
            # Set up API Client
            key = os.environ.get("OPENAI_API_KEY")
            self.api_client = OpenAI(api_key = key)
        elif api_name == "gemini":
            key = os.environ.get("GEMINI_API_KEY")
            self.api_client = genai.Client(api_key=key)
        else:
            ValueError(f"{api_name} is not a valid LLM API")
        self.api_name = api_name 
        self.help_rate = help_rate

        # Set up data stream
        self.image_deque = deque(maxlen = 15) # keep track of last 15 images, sampling 3x per second (5 seconds)
        self.position_deque = deque(maxlen = 15)
        self.obs_rate = obs_rate

        # Set up map info 
        self.goal_pose = goal_pose
        self.goal_pose_utm = utm.from_latlon(goal_pose[0], goal_pose[1]) 
        self.marker_size = 10

        tiler = cimgt.OSM()
        tiler._executor = None # so it doesn't do multithraeding so that it plays nice later 
        crs = tiler.crs

        self.fig, self.ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={"projection": crs}
        )
        self.ax.set_extent([map_bounds[0][1], map_bounds[1][1] , 
                    map_bounds[0][0], map_bounds[1][0]], 
                    crs=ccrs.PlateCarree())

        self.ax.add_image(tiler, 18)  # Add OpenStreetMap tiles; zoom level lower values = less zoomed-in
        
        self.ax.plot( # Plot goal position 
                self.goal_pose[1], self.goal_pose[0],
                marker="o", color="red", markersize=self.marker_size, 
                transform=ccrs.PlateCarree()
        )
        self.no_pos = True

        self.map_lock = threading.Lock()


        self.map_angle_annotations =  { # with 0 being up, 90 being left 
            "forward": (np.radians(0), "green"),           
            "forward_right": (np.radians(-45), "yellow"),    
            "forward_left": (np.radians(45), "blue"),    
            "right": (np.radians(-90), "orange"),             
            "left": (np.radians(90), "indigo"),              
            "back": (np.radians(180), "pink"),          
            "back_right": (np.radians(135), "red"),     
            "back_left": (np.radians(225), "violet"),        
        }

        self.image_angle_annotations_pil = Image.open("directions overlay.png").convert("RGBA")


    def get_help_prompt(self, rel_goal_pos):

        prompt =  "I am a wheeled robot, and you want to help me navigate to my desired goal position. I do not want to crash, enter streets that cars drive on, or go down stairs."

        prompt += "I am giving you a birds eye view map of my environment. The blue dots represent positions I have visted. The red point is my eventual goal."

        prompt += "My latest position is marked with the cluster of arrows indicating where I can go from here."


        for direction in self.map_angle_annotations.keys():
            angle_offset, color = self.map_angle_annotations[direction]
            prompt += f"On both the latest image observation and the birds eye view map, going in {direction} is marked with a {color} arrow"   


        prompt += f"At the moment, my final goal is {rel_goal_pos[0]} meters forward and {rel_goal_pos[1]} to the left of me."

        prompt += """What general direction should I go, in my frame of reference, to get closer to the red goal?
                    Don't use cardinal directions. Instead, tell me broadly to go right / left / straight / forward / backwards or a combination of those. 
        """

        return prompt 


    def get_robot_data(self):
        last_run = time.time()  # Track when the last iteration ran
        interval = 1 / self.obs_rate

        while True:
            current_time = time.time()
            if current_time - last_run < interval:
                continue # Not ready yet 

            last_run = current_time 

            # Get most recent data from robot
            screenshot = requests.get("http://127.0.0.1:8000/v2/screenshot")
            screenshot = screenshot.json() 
            self.image_deque.append(screenshot["front_frame"])
            
            # Get GPS data
            gpsdata = requests.get("http://127.0.0.1:8000/data")  
            gpsdata = gpsdata.json() 
            self.position_deque.append((gpsdata["latitude"], gpsdata["longitude"], gpsdata["orientation"]))

            rad_orientation = np.radians(convert_angle_compass_to_cartesian(gpsdata["orientation"]))

            # Update map
            offset_length = 30
            self.map_lock.acquire()
            try:
                if not self.no_pos:
                # Attempt 1 : History dots, arrow orientation 
                #     for artist in reversed(self.ax.get_children()): # Make last pos a history point instead 
                #         if isinstance(artist, matplotlib.quiver.Quiver):  # Check if it's a quiver
                #             xdata, ydata = artist.U, artist.V  # Get quiver's vector components
                #             artist.remove()  # Remove the old quiver

                #             # Plot a history point at the quiver's location
                #             self.ax.plot(
                #                 artist.X, artist.Y, 
                #                 marker="o", color="blue", markersize=self.marker_size // 2,
                #                 transform=ccrs.PlateCarree()
                #             )
                #             break  # Stop after modifying the most recent one

                # arrow_length = 50 
                # dx = arrow_length * np.cos(rad_orientation)
                # dy = arrow_length * np.sin(rad_orientation)

                # self.ax.quiver(
                #     np.array([gpsdata["longitude"]]), np.array([gpsdata["latitude"]]),  # Start position
                #     np.array([dx]), np.array([dy]),  # Direction vector
                #     angles='xy', scale_units='xy', scale=1, color="green",
                #     transform=ccrs.PlateCarree()
                # )

                # Attempt 2: history dots, arrows for each possible direction
                    # Clear old quivers 
                    remove_left = len(self.map_angle_annotations.keys())
                    for artist in reversed(self.ax.get_children()): # Remove old quivers
                        if isinstance(artist, matplotlib.quiver.Quiver):  # Check if it's a quiver
                            xdata, ydata = artist.U, artist.V  # Get quiver's vector components
                            artist.remove()  # Remove the old quiver

                            remove_left -= 1

                            if remove_left == 0:
                                self.ax.plot(
                                    artist.X, artist.Y, 
                                    marker="o", color="blue", markersize=self.marker_size // 2,
                                    transform=ccrs.PlateCarree()
                                )
                                break  
                
                    for direction in self.map_angle_annotations.keys():
                        angle_offset, color = self.map_angle_annotations[direction]
                        total_angle = rad_orientation + angle_offset
                        dx_offset = offset_length * np.cos(total_angle)
                        dy_offset = offset_length * np.sin(total_angle)
                        self.ax.quiver(
                            np.array([cur_pos[1]]), np.array([cur_pos[0]]),  # Start position
                            np.array([dx_offset]), np.array([dy_offset]),  # Direction vector
                            angles='xy', scale_units='xy', scale=1.5, color= color,
                            transform=ccrs.PlateCarree()
                        )

                self.no_pos = False
            finally:
                self.map_lock.release()

    def get_api_help(self):
        last_run = time.time()  # Track when the last iteration ran
        interval = 1 / self.help_rate

        while True:
            current_time = time.time()
            if current_time - last_run < interval:
                continue # Not ready yet 
            
            last_run = current_time 

            # Get latest map 
            img_type_map = "image/png"
            self.map_lock.acquire()
            try:
                plt.savefig("./current_map.png", dpi=300, bbox_inches="tight")
            finally:
                self.map_lock.release()
            img_b64_str_map = encode_image_to_base64("./current_map.png")

            # Get image history
            img_type_obs = "image/jpeg"
            img_history = []
            for i in range(-1, -len(self.image_deque), - self.obs_rate): # get 1 img per second
                img_history.append(self.image_deque[i])
            
            if len(img_history) < 3:
                continue  # wait for more observations 
            
            # Annotate the latest image observation
            img_history[0] = pil_to_byte(overlay_imgs(byte_to_pil(img_history[0]), self.image_angle_annotations_pil))


            # Get latest position & Goal pose 
            gpsdata = self.position_deque[-1] 
            cur_compass = cur_compass = -float(gpsdata[2])/180.0*3.141592 # don't reorient to have 0 as north 
            cur_utm = utm.from_latlon(gpsdata[0], gpsdata[1]) 
            del_x, del_y = calculate_relative_position(cur_utm[0], cur_utm[1], 
                                                       self.goal_pose_utm[0], self.goal_pose_utm[1])
            print("Del x", del_x, "del y", del_y, "compass", cur_compass)
            rel_goal_pos = rotate_to_local_frame(del_x, del_y, cur_compass)
            print("rel goal pos", rel_goal_pos)
            rel_goal_pos = [rel_goal_pos[1], -rel_goal_pos[0]] # change it to be forward, left 
            print("relative goal pose is ", rel_goal_pos )
            
            # Get help prompt
            help_prompt = self.get_help_prompt(rel_goal_pos)

            # Get suggestion 
            self.api_start_time = time.time()
            if self.api_name == "GPT":
                content = [{"type": "text", "text": help_prompt}, 
                       {"type": "image_url",
                        "image_url": {"url": f"data:{img_type_map};base64,{img_b64_str_map}"}
                        }]
                
                img_history = resize_images_byte(img_history)
                
                content.extend([{
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type_obs};base64,{img}"},
                    } for img in img_history])


                response = self.api_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    max_tokens = 1000,
                )
                print(response.choices[0].message.content)
            elif self.api_name == "gemini":

                input_images = [img_b64_str_map] + img_history
                input_images = resize_images_byte(input_images)

                contents = [help_prompt] +  input_images

                response = self.api_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=contents)

                print(response.text)

            print(f"response took {time.time() - self.api_start_time} seconds to generate \n")


    def run(self):
        data_thread = threading.Thread(target=self.get_robot_data, daemon = False)
        api_thread = threading.Thread(target=self.get_api_help, daemon = False)

        data_thread.start()
        api_thread.start()

        data_thread.join()
        api_thread.join()


if __name__ == "__main__":
    cur_pos = np.array([37.873550 , -122.267617])
    campanile_pos = np.array([37.87210, -122.25780])

    points = np.vstack([cur_pos, campanile_pos])
    min_lat, min_long = np.min(points, axis = 0)
    max_lat, max_long = np.max(points, axis = 0)

    border = 0.001
    bounds = [[min_lat - border, min_long - border],
            [max_lat + border, max_long + border]]


    LLMHelper(
        map_bounds = bounds,
        goal_pose = campanile_pos,
        obs_rate = 3,
        help_rate = 1,
        api_name = "GPT",
    ).run()
