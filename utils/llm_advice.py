import threading
import time
import requests

import numpy as np 
from collections import deque

import re
import os
import sys
import utm 
import math 
import select 
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


GOAL_THRES = 20 # 10 meters away - counts as reaching the goal! 

import utils 

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
        self.map = utils.Map(map_bounds)
        self.map.add_grid(6, 15)
        self.map_lock = threading.Lock()

        # Goal Info 
        self.goal_pose_final = goal_pose
        self.goal_stack = deque()
        self.goal_stack.append(self.goal_pose_final)

       
        while len(self.position_deque) == 0:
             self.get_robot_data() # Need to have the current position to plan trajectory! 

        intermediate_goals = None
        while intermediate_goals is None:
                intermediate_goals = self.get_intermediate_goals(self.goal_pose_final)


        
        for i in range(2, -1, -1): # just get the closest 3 - recompute once we're there. 
            self.goal_stack.append(self.map.coords_to_lat_long(
                                                intermediate_goals[i][0],
                                                intermediate_goals[i][1] ))

        print("First goal at", self.goal_stack[-1][0],self.goal_stack[-1][1] )
        

    def get_robot_data_loop(self):
        last_run = time.time()  # Track when the last iteration ran
        interval = 1 / self.obs_rate

        while True:
            current_time = time.time()
            if current_time - last_run < interval:
                continue # Not ready yet 

            last_run = current_time 

            self.get_robot_data()


    def get_robot_data(self):
        # Get most recent data from robot
        screenshot = requests.get("http://127.0.0.1:8000/v2/screenshot")
        screenshot = screenshot.json() 
        self.image_deque.append(screenshot["front_frame"])
        
        # Get GPS data
        robotdata = requests.get("http://127.0.0.1:8000/data")
        robotdata = robotdata.json() 

        # if robotdata["latitude"] == 1000:
        #     print("Lost GPS signal, got 1000")
        #     return 
        
        gpsdata = requests.get("http://127.0.0.1:3000/last-data")
        gpsdata = gpsdata.json()
        # gpsdata = robotdata 

        new_pos = (gpsdata["latitude"], gpsdata["longitude"], robotdata["orientation"])
        # self.position_deque.append(new_pos)

        # Add GPS data to map history if it isn't the same as before 
        if len(self.position_deque) < 2 or self.position_deque[-1] != new_pos:
            self.position_deque.append(new_pos)
            self.map_lock.acquire()
            try:
                self.map.plot_points([new_pos[:2]], "cyan", size=7)
            finally:
                self.map_lock.release()


    def latest_goal_dist(self):

        cur_pos = self.position_deque[-1] 
        cur_compass = -float(cur_pos[2])/180.0*3.141592 

        cur_utm = utm.from_latlon(cur_pos[0], cur_pos[1]) 
        goal_utm = utm.from_latlon(self.goal_stack[-1][0], self.goal_stack[-1][1]) 

        del_x, del_y = utils.calculate_relative_position(cur_utm[0], cur_utm[1], 
                                                   goal_utm[0], goal_utm[1])
        
        return math.sqrt(del_x ** 2 + del_y ** 2)


    def get_intermediate_goals(self, goal_pos, spacing=50, was_stuck = False):

        # Figure out how many waypoints we want
        cur_pos = self.position_deque[-1]
        dist = utils.dist_between_latlon(cur_pos, goal_pos)
        num = dist // spacing + 1 

        # Set up prompt
        goal_prompt = f"""I am a small wheeled root trying to navigate an environment. In this overhead map, my position is marked with a bright fuchsia dot, locations I have visited are marked with light blue dots, red dots indicate spots I have gotten stuck at or couldn't make progress from such as stairs or dead ends, and a purple dot represents my goal position. I am small compared to the size of the plotted dots, so I can go through narrow areas on the map. The map is split into a coordinate grid, with labels along the x (bottom) and y (vertical) axis. 
        
        First, find the maximum x value and the maximum y value. What are those bounds on my map? 
        
        Next, what is my approximate current position in the coordinate grid? What is the approximate goal position?  Use precision up to 3 decimal spots, such as 2.568. 
        
        Now, give me a path from my current position to the goal position using {num} intermediate waypoints. I am a wheeled robot, so I *must* stay on roads (white, gray, or light red areas). Traveling through buildings (marked in brown) is strictly forbidden and will cause me to get stuck.

        The path you provide *must* be a smooth continuous sequence of waypoints that a wheeled robot can realistically follow. You cannot teleport from one path to an adjacent one. Each waypoint must be reachable from the previous waypoint *without* passing through any buildings or off-road areas. This means you might sometimes need to move a little bit further away from the goal to ultimately make progress. 
        
        Format the path like this [(x, y), (x, y), ..., (x, y)].  Use precision up to 3 decimal spots, such as 2.568. Don't include the start pose or the final goal pose in the path."""

        if was_stuck:
            goal_prompt += "I am currently STUCK in my position and I haven't made progress in 30 seconds, so you especially should consider a creative path going around buildings instead of taking a straighter approach."

        # Set up map (plot cur pos, goal pos)
        self.map_lock.acquire()
        try:
            self.map.plot_points([cur_pos[:2]], "fuchsia", size=9)
            self.map.plot_points([goal_pos[:2]], "blueviolet", size=9)
            self.map.save_map("./goals.png")
            self.map.remove_points(1, "fuchsia")
            self.map.remove_points(1, "blueviolet")
        finally:
            self.map_lock.release()

        img_type_map = "image/png"
        img_b64_str_map = utils.encode_image_to_base64("./goals.png")
        img_b64_str_map = utils.resize_image_byte(img_b64_str_map)

        # Send Query
        api_start_time = time.time()
        if self.api_name == "GPT":
            content = [{"type": "text", "text": goal_prompt}, 
                    {"type": "image_url",
                    "image_url": {"url": f"data:{img_type_map};base64,{img_b64_str_map}"}
                    }]
            
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
            ans = response.choices[0].message.content
        elif self.api_name == "gemini":

            contents = []
            contents.append({"role": "user",
                                "parts": [
                                    {
                                        "text": goal_prompt,
                                    },
                                    {
                                        "inline_data" : {
                                            "mime_type" : "image/png",
                                            "data": img_b64_str_map
                                        }
                                    }
                                ]
                    })
            
            response = self.api_client.models.generate_content(
                model="gemini-2.0-flash", # not -exp
                contents=contents)
            ans = response.text
        print(f"Initial response took {time.time() - api_start_time} seconds to generate \n")
        print("full answer", ans)

        goals = utils.get_tuples_list_floats(ans) # extract out waypoints
        print("\n\nGot Goals", goals)

        # Skip first waypoint if it was generated too close to the current position 
        first_goal = goals[0]
        if utils.dist_between_latlon(self.map.coords_to_lat_long(first_goal[0], first_goal[1]), cur_pos) < GOAL_THRES + 1:
            goals = goals[1:] 

        # Refine goals with visual feedback 
        goal_refine_prompt = f""" I originally asked you this: \n {goal_prompt}
        Here is your predicted path, {goals}, displayed with orange dots. With this information, adjust the waypoints, making sure that the path is continuous and does not go through buildings.
        """

        # Annotate map with predicted goals & then restore the map to just history 
        self.map_lock.acquire()
        try:
            self.map.plot_points([cur_pos[:2]], "fuchsia", size=9)
            self.map.plot_points([goal_pos[:2]], "blueviolet", size=9)
            self.map.plot_points(goals, "orange", False, size=9)
            # self.map.plot_points(goals, [plt.cm.viridis(i / (len(goals)-1)) for i in range(len(goals))], False, size=9)
            self.map.save_map("./goals.png")
            self.map.remove_points(1, "fuchsia")
            self.map.remove_points(1, "blueviolet")
            self.map.remove_points(len(goals), "orange")
        finally:
            self.map_lock.release()

        img_type_map = "image/png"
        img_b64_str_map = utils.encode_image_to_base64("./goals.png")
        img_b64_str_map = utils.resize_image_byte(img_b64_str_map)

        api_start_time = time.time()
        if self.api_name == "GPT":
            content = [{"type": "text", "text": goal_refine_prompt}, 
                    {"type": "image_url",
                    "image_url": {"url": f"data:{img_type_map};base64,{img_b64_str_map}"}
                    }]
            
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
            ans = response.choices[0].message.content
        elif self.api_name == "gemini":
            contents = []
            contents.append({"role": "user",
                                "parts": [
                                    {
                                        "text": goal_refine_prompt,
                                    },
                                    {
                                        "inline_data" : {
                                            "mime_type" : "image/png",
                                            "data": img_b64_str_map
                                        }
                                    }
                                ]
                    })
            
            response = self.api_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents)
            ans = response.text
        print(f"Refined response took {time.time() - api_start_time} seconds to generate \n")
        print("full refined answer", ans)

        goals = utils.get_tuples_list_floats(ans) # extract out waypoints
        return goals 
    
    def replan(self, goal, spacing, keep_num):

        intermediate_goals = None
        while intermediate_goals is None:
            intermediate_goals = self.get_intermediate_goals(self.goal_pose_final, spacing = 50)

        self.goal_stack.pop()

        for i in range(keep_num - 1, -1, -1): # just save the closest num 
            self.goal_stack.append(self.map.coords_to_lat_long(
                                    intermediate_goals[i][0],
                                    intermediate_goals[i][1] ))
        

        print(f"New goal at {self.goal_stack[-1][0]:.8f} , {self.goal_stack[-1][1]:.8f}" )




    def runner(self):

        # Keep track of distance from goal 
        dist_check = 120 # if we haven't made progress (2 meters) in 120 seconds, that's a issue 
        dist_update = 1
        dist_thres = 2
        last_check = time.time()
        last_update = time.time()
        latest_distances = deque(maxlen = dist_check)
        latest_distances.append(self.latest_goal_dist())

        num_goals_before_replan = 1


        just_stuck = False

        while True:
            # get feedback on being stuck from robot
            i, o, e = select.select([sys.stdin], [], [], 0.001)
            if i:
                received_input = sys.stdin.readline().strip()
                if "unsafe" in received_input:

                    # Mark spot as UNSAFE 
                    self.map_lock.acquire()
                    try:
                        self.map.plot_points([self.position_deque[-1][:2]], "red", size=12)
                    finally:
                        self.map_lock.release()

                print("recieved input", received_input)


            cur_dist = self.latest_goal_dist()

            # Keep track of distance from goal 
            if time.time() - last_update > dist_update:
                latest_distances.append(cur_dist)
                print(f"now {cur_dist} away ")
                last_update = time.time()
            
            # Check if we've reached the closest goal 
            if cur_dist < GOAL_THRES:
                self.goal_stack.pop()
                num_goals_before_replan -= 1

                print(f"Goal reached! Next goal is {self.goal_stack[-1][0]:.8f} , {self.goal_stack[-1][1]:.8f}")

                if num_goals_before_replan == 0:
                    print("Replanning")
                    self.replan(self.goal_pose_final, spacing = 50, keep_num = 3)
                    
                    num_goals_before_replan = len(self.goal_stack) - 2
                    latest_distances.clear()
                    cur_dist = self.latest_goal_dist()
                    latest_distances.append(cur_dist)
                    last_update = time.time()

            # Check if we're stuck 
            if time.time() - last_check > dist_check and cur_dist + dist_thres >= latest_distances[0]:
                print(f"No progress made in {dist_check} seconds, used to be {latest_distances[0]} away and is now {cur_dist} away")

                print("STUCK, pause operation for now")

                # Fully replan 
                self.goal_stack.clear()
                self.goal_stack.append(self.goal_pose_final)

                self.replan(self.goal_pose_final, spacing = 50, keep_num = 3)
                    
                num_goals_before_replan = len(self.goal_stack) - 2
                latest_distances.clear()
                cur_dist = self.latest_goal_dist()
                latest_distances.append(cur_dist)
                last_update = time.time()
                
                print(f"New goal at {self.goal_stack[-1][0]:.8f} , {self.goal_stack[-1][1]:.8f} resume operation ")

    def run(self):
        data_thread = threading.Thread(target=self.get_robot_data_loop, daemon = True)
        main_thread = threading.Thread(target=self.runner, daemon = True)

        data_thread.start()
        main_thread.start()

        data_thread.join()
        main_thread.join()


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
        api_name = "gemini",
    ).run()
