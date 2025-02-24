import threading
import time
import requests

import numpy as np 
from collections import deque

import os
import base64

from openai import OpenAI

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use("Agg")  # Use a non-interactive backend to work with multithreading


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        img_b64_bytes = base64.b64encode(image_file.read())
        img_b64_str = img_b64_bytes.decode("utf-8")
    return img_b64_str


help_prompt = f"""
I am a wheeled robot, and you want to help me navigate to my desired goal position. I do not want to crash or enter streets that cars drive on. 

I am giving you a birds eye view map of my environment. The blue dots represent positions I have visted. The green dot is my current position. The red point is my eventual goal.

I am also giving you the image observations from my camera for the last 5 seconds.

Based on this information, give me a general direction I should go in to make progress towards my goal, such as "go to the right", or "go to the trash can" or "back up".

It should be brief, and for these specific circumstances. 
"""


class LLMHelper():

    def __init__(self, map_bounds, goal_pose, obs_rate, help_rate):
        # Set up API Client
        key = os.environ.get("OPENAI_API_KEY")
        self.api_client = OpenAI(api_key = key)
        self.help_rate = help_rate

        # Set up data stream
        self.image_deque = deque(maxlen = 15) # keep track of last 15 images, sampling 3x per second (5 seconds)
        self.position_deque = deque(maxlen = 15)
        self.obs_rate = obs_rate

        # Set up map info 
        self.goal_pose = goal_pose

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
                marker="o", color="red", markersize=marker_size, 
                transform=ccrs.PlateCarree()
        )

        self.ax.plot( # Plot extra point to remove as "start position"
                self.goal_pose[1], self.goal_pose[0],
                marker="o", color="green", markersize=marker_size, 
                transform=ccrs.PlateCarree()
        )

        self.map_lock = threading.Lock()


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

            # Update map
            self.map_lock.acquire()
            try:
                for artist in reversed(self.ax.get_children()): # Make last pos a history point instead 
                    if isinstance(artist, matplotlib.lines.Line2D) and artist.get_marker() == "o":
                        artist.set_color("blue")  # Change latest to be a history point instead
                        artist.set_markersize(marker_size // 2)
                        break

                self.ax.plot( # Plot new current position 
                    gpsdata["longitude"], gpsdata["latitude"],
                    marker="o", color="green", markersize=marker_size,
                    transform=ccrs.PlateCarree()
                )
                print("map updated ")
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
            img_info = []
            for i in range(-1, -len(self.image_deque), - self.obs_rate): # get 1 img per second
                img_info.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_type_obs};base64,{self.image_deque[i]}"},
                })

            # Create context
            content = [{"type": "text", "text": help_prompt}, 
                       {"type": "image_url",
                        "image_url": {"url": f"data:{img_type_map};base64,{img_b64_str_map}"}
                        }]
            content.extend(img_info)

            # Get suggestion 
            self.api_start_time = time.time()
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
            print(f"response took {time.time() - self.api_start_time} seconds to generate")


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

    marker_size = 10


    LLMHelper(
        map_bounds = bounds,
        goal_pose = campanile_pos,
        obs_rate = 3,
        help_rate = 1,
    ).run()
