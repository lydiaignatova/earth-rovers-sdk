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

import sys
import time
import math
import json
import base64
import argparse
import yaml
import requests
from collections import deque
from typing import List, Tuple, Dict, Optional
from io import BytesIO

import torch
import numpy as np
import utm
from PIL import Image
from torchvision import transforms

# Set up path for custom modules
DIR_loc = "/home/lydia/Documents/work/rail/repos/Learning-to-Drive-Anywhere-via-MBRA"
sys.path.insert(0, f"{DIR_loc}/train")

from vint_train.models.il.il import IL_dist, IL_gps


GOAL_THRES = 20 # 10 meters away - counts as reaching the goal! 

import utils 

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths
MODEL_WEIGHTS_PATH = f"{DIR_loc}/policy/MBL_based_gps"
MODEL_CONFIG_PATH = f"{DIR_loc}/train/config"

# Constants
METRIC_WAYPOINT_SPACING = 0.25
THRES_DIST = 30.0
THRES_UPDATE = 5.0

MAX_V = 0.3
MAX_W = 0.3

RATE = 3.0
EPS = 1e-8  # Default value of NoMaD inference
DT = 1 / 4  # Default value of NoMaD inference


def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi

def decode_from_base64(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    return image

def transform_images_exaug(pil_imgs: List[Image.Image]) -> torch.Tensor:
    """Transforms a list of PIL image to a torch tensor."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size

        transf_img = transform_type(pil_img)
        transf_img = torch.unsqueeze(transf_img, 0)
        transf_imgs.append(transf_img)
    return torch.cat(transf_imgs, dim=1)

def calculate_relative_position(x_a, y_a, x_b, y_b):
    delta_x = x_b - x_a
    delta_y = y_b - y_a
    return delta_x, delta_y

# Rotate the relative position to the robot's local coordinate system
def rotate_to_local_frame(delta_x, delta_y, heading_a_rad):
    # Apply the rotation matrix for the local frame
    relative_x = delta_x * math.cos(heading_a_rad) + delta_y * math.sin(heading_a_rad)
    relative_y = -delta_x * math.sin(heading_a_rad) + delta_y * math.cos(heading_a_rad)
    
    return relative_x, relative_y


class Actor():

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

        # Set up Model
        # Set up model 
        model_config_path = MODEL_CONFIG_PATH + "/" + "frodobot_dist_IL2_gps.yaml"    
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)

        self.context_size = config["context_size"]

        self.model = IL_gps(
                context_size=config["context_size"],
                len_traj_pred=config["len_traj_pred"],
                learn_angle=config["learn_angle"],
                obs_encoder=config["obs_encoder"],
                obs_encoding_size=config["obs_encoding_size"],
                late_fusion=config["late_fusion"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )   
        
        model_path = MODEL_WEIGHTS_PATH + "/" + "latest.pth"
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.to(device)
        print("Model Loaded ")

        self.pause = False 


        # Set up data stream
        self.image_deque_byte = deque(maxlen = 15) # keep track of last 15 images, sampling 3x per second (5 seconds)
        self.image_deque_pil = deque(maxlen = 15)
        self.position_deque = deque(maxlen = 15)
        self.obs_rate = obs_rate

        # Set up map info 
        self.map = utils.Map(map_bounds)
        self.map.add_grid(6, 15)
        self.map_lock = threading.Lock()

        # Goal Info 
        self.goal_pose_final = goal_pose
        self.goal_stack = deque() # Goals are stored as lat, long 

        self.goal_stack.append(self.goal_pose_final)
        self.goal_lock = threading.Lock()

        while len(self.position_deque) == 0:
             self.get_robot_data() # Need to have the current position to plan trajectory! 

        self.replan(self.goal_pose_final, spacing=50, keep_num= 3)

        print("First goal at", self.goal_stack[-1][0],self.goal_stack[-1][1] )

        # Intervention info 

        self.correction_to_action = { # linear, angular, frequency
            "back0": [(-1., 0., 3)],
            "back45left": [(-1., 0., 2), (-1., -1., 1)],
            "back45right": [(-1., 0., 2), (-1., 1., 1)],
            "back90left": [(-1., 0., 2), (-0.5, -1., 3)],
            "back90right": [(-1., 0., 2), (-0.5, 1., 3)],
            "back180": [(-1., 0., 2), [0.5, 1., 6]],
            "front45left": [(1., 0., 2), (1., 1., 1)],
            "front45right": [(1., 0., 2), (1., -1., 1)],
            "front90left": [(1., 0., 2), (0.5, 1., 3)],
            "front90right": [(1., 0., 2), (0.5, -1., 3)],
            "front180": [(1., 0., 2), (0.5, -1., 6)],
            "back180": [(-1., 0., 2), (0.5, -1., 6)],
        }

        self.corrections_idx = [
            "back0",
            "back45left",
            "back45right",
            "back90left",
            "back90right",
            "back180",
            "front45left",
            "front45right",
            "front90left",
            "front90right", 
            "front180",
            "back180",
        ]
     

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
        self.image_deque_byte.append(screenshot["front_frame"])

        img_PIL = decode_from_base64(screenshot["front_frame"])
        img_PIL = img_PIL.convert('RGB')

        self.image_deque_pil.append(img_PIL)
        
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
        goal_prompt = f"""I am a small wheeled root trying to navigate an environment. In this overhead map, my position is marked with a bright fuchsia dot, locations I have visited are marked with light blue dots, red dots indicate spots I have gotten stuck at or couldn't make progress from, and a purple dot represents my goal position. I am small compared to the size of the plotted dots, so I can go through narrow areas on the map. The map is split into a coordinate grid, with labels along the x (bottom) and y (vertical) axis. 
        
        First, find the maximum x value and the maximum y value. What are those bounds on my map? 
        
        Next, what is my approximate current position in the coordinate grid? What is the approximate goal position?  Use precision up to 3 decimal spots, such as 2.568. 
        
        Now, give me a path from my current position to the goal position using {num} intermediate waypoints. For each waypoint, give me the x and y position on the coordiante grid, as well as the orientation, theta, where the robot should be facing at that waypoint. North (up on the map) is 0, West (left on the map) is 90, South (down on the map) is 180, and East (right on the map) is 270. You can choose intermediate orientations, too, like 210 for mostly South, and a bit to the East. I am a wheeled robot, so I *must* stay on roads (white, gray, or light red areas). Traveling through buildings (marked in brown) is strictly forbidden and will cause me to get stuck. Also try to minimize crossing big white roads, as cars might hit me.

        The path you provide *must* be a smooth continuous sequence of waypoints that a wheeled robot can realistically follow. Each waypoint must be reachable from the previous waypoint *without* passing through any buildings or off-road areas. This means you might sometimes need to move further away from the goal to ultimately make progress. 
        
        Format the path like this [(x, y, theta), (x, y, theta), ..., (x, y, theta)].  Use precision up to 3 decimal spots, such as 2.568. You do not need to stay on the grid lines. Don't include the start pose or the goal pose in the path."""

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

        goals = utils.get_tuples_list_floats_3(ans) # extract out waypoints
        print("\n\nGot Goals", goals)

        # Skip first waypoint if it was generated too close to the current position 
        first_goal = goals[0]
        if utils.dist_between_latlon(self.map.coords_to_lat_long(first_goal[0], first_goal[1]), cur_pos) < GOAL_THRES + 1:
            goals = goals[1:] 
        goals = np.array(goals)

        # Refine goals with visual feedback 
        goal_refine_prompt = f""" I originally asked you this: \n {goal_prompt}
        Here is your predicted path, {goals}, displayed with orange dots. With this information, adjust the waypoints, making sure that the path is continuous and does not go through buildings.
        """

        # Annotate map with predicted goals & then restore the map to just history 
        self.map_lock.acquire()
        try:
            self.map.plot_points([cur_pos[:2]], "fuchsia", size=9)
            self.map.plot_points([goal_pos[:2]], "blueviolet", size=9)
            self.map.plot_points(goals[:, :2], "orange", False, size=9)
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

        goals = utils.get_tuples_list_floats_3(ans) # extract out waypoints
        return goals 


    def replan(self, goal, spacing, keep_num):

        intermediate_goals = None
        while intermediate_goals is None:
            intermediate_goals = self.get_intermediate_goals(goal, spacing = spacing)

        self.goal_stack.clear()
        self.goal_stack.append(self.goal_pose_final)

        for i in range(keep_num - 1, -1, -1): # just save the closest num 
            lat, long = self.map.coords_to_lat_long(
                                    intermediate_goals[i][0],
                                    intermediate_goals[i][1])
            orientation = intermediate_goals[i][2] / 180 * 3.1415
            self.goal_stack.append([lat, long, orientation])
        
        print(f"Newest goal at {self.goal_stack[-1][0]:.8f} , {self.goal_stack[-1][1]:.8f}" )


    def action_from_waypoint(self, dx, dy, hx, hy):
        if np.abs(dx) < EPS and np.abs(dy) < EPS:
            lin_vel = 0
            ang_vel = clip_angle(np.arctan2(hy, hx))/DT
        elif np.abs(dx) < EPS:
            lin_vel =  0
            ang_vel = np.sign(dy) * np.pi/(2*DT)
        else:
            lin_vel = dx / DT
            ang_vel = np.arctan(dy/dx) / DT
            
        lin_vel = np.clip(lin_vel, 0, 0.5)
        ang_vel = np.clip(ang_vel, -1.0, 1.0)

        # If just angular motion, scale up to have enough to make a difference 
        if lin_vel < 0.05 and np.absolute(ang_vel) < 0.2 and np.absolute(ang_vel) > 0.05:
            ang_vel = np.sign(ang_vel)*0.2
            lin_vel = lin_vel*0.2/np.absolute(ang_vel)
            
        return lin_vel, ang_vel


    def compute_action(self):

        linear_vel = None
        angular_vel = None
        distances = None
        waypoints = None

        lat, long, orientation = self.position_deque[-1]
        cur_gps = (lat, long)
        cur_utm = utm.from_latlon(lat, long)
        cur_compass = -orientation / 180 * 3.1415  # Convert from CW 

        goal = self.goal_stack[-1]
        goal_gps = goal[0], goal[1]
        goal_utm = utm.from_latlon(goal_gps[0], goal_gps[1])
        goal_compass = goal[2] / 180 * 3.1415 # already CCW
        

        if len(self.image_deque_pil) < self.context_size + 1:
            return 0, 0 # no action while not enough context 
        
        else:
            latest_imgs = list(self.image_deque_pil)[-(self.context_size+1):]
            input_imgs = transform_images_exaug(latest_imgs)
            input_imgs = input_imgs.to(device)

            delta_x, delta_y = calculate_relative_position(cur_utm[0], cur_utm[1], goal_utm[0], goal_utm[1])
            relative_x, relative_y = rotate_to_local_frame(delta_x, delta_y, cur_compass)
            
            # If goal is too far (past THRES_DIST), project it closer 
            if np.sqrt(relative_x**2 + relative_y**2) > THRES_DIST:
                relative_x = relative_x/np.sqrt(relative_x**2 + relative_y**2)*THRES_DIST
                relative_y = relative_y/np.sqrt(relative_x**2 + relative_y**2)*THRES_DIST   
            
            goal_pose = np.array([relative_y/METRIC_WAYPOINT_SPACING, -relative_x/METRIC_WAYPOINT_SPACING, np.cos(goal_compass-cur_compass), np.sin(goal_compass-cur_compass)])
            goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
            
            print("\nCurrent position", cur_gps, "orientation", cur_compass)
            print("Goal position", goal_gps[0], ",", goal_gps[1])
            print("Relative pose", goal_pose[0]*METRIC_WAYPOINT_SPACING, goal_pose[1]*METRIC_WAYPOINT_SPACING, goal_pose[2], goal_pose[3])

            with torch.no_grad():  
                waypoints = self.model(input_imgs, goal_pose_torch)                 

            waypoints = waypoints.cpu().detach().numpy()
            chosen_waypoint = waypoints[0][2].copy() # select waypoint 2 according with NoMaD 

            chosen_waypoint[:2] *= (MAX_V / RATE) # normalize 
            dx, dy, hx, hy = chosen_waypoint

            lin, ang = self.action_from_waypoint(dx, dy, hx, hy)
            return lin, ang 
           

    def take_action(self, linear, angular):
        data = json.dumps({"command": {"linear": float(linear), "angular": float(angular)}})
        response = requests.post("http://127.0.0.1:8000/control", data=data)
        response = response.json()
        if response["message"] == 'Command sent successfully':
            print(response["message"], "linear", linear, "angular", angular)
            return True 
        else:
            return False

    def get_correction(self):

        correction_prompt = f"""
        I am a small wheeled root trying to navigate an environment. In this overhead map, my position is marked with a bright fuchsia dot, locations I have visited are marked with light blue dots, red dots indicate spots I have gotten stuck at or couldn't make progress from, and a purple dot represents my goal position. I am small compared to the size of the plotted dots, so I can go through narrow areas on the map. The map is split into a coordinate grid, with labels along the x (bottom) and y (vertical) axis. 

        I have stopped being able to make progress towards my desired goal. You want to help me get back on track. I am also giving you my most recent forward facing image observation. 

        My options for interventions to get back on track are:
        0. Back straight up
        1. Back up and turn 45 degrees to the right
        2. Back up and turn 45 degrees to the left
        3. Back up and turn 90 degrees to the right
        4. Back up and turn 90 degrees to the left
        5. Back up and turn 180 degrees
        6. Move forward and turn 45 degrees to the right
        7. Move forward and turn 45 degrees to the left
        8. Move forward and turn 90 degrees to the right
        9. Move forward and turn 90 degrees to the left
        10. Move forward and turn 180 degrees
        11. Move backward and turn 180 degrees

        In the absolute worst case, I can call for human intervention, which you can reference with index 12, but this should be reserved for extreme cases, such as if I have fallen.

        Please respond to this message in the following format:
        "Suggested intervention: [Pick the index of the most appropriate intervention];
        Reason for intervention: [Explanation of why the chosen index is a good approach]"
        """

        # Set up map (plot cur pos, goal pos)
        cur_pos = self.position_deque[-1]
        goal_pos = self.goal_pose_final

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

        img_type_obs = "image/jpeg"
        img_b64_obs = utils.resize_image_byte(self.image_deque_byte[-1])
        
        # Send Query
        api_start_time = time.time()
        if self.api_name == "GPT":
            content = [{"type": "text", "text": correction_prompt}, 
                    {"type": "image_url",
                    "image_url": {"url": f"data:{img_type_map};base64,{img_b64_str_map}"}
                    },
                    {"type": "image_url",
                    "image_url": {"url": f"data:{img_type_obs};base64,{img_b64_obs}"}
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
                                        "text": correction_prompt,
                                    },
                                    {
                                        "inline_data" : {
                                            "mime_type" : img_type_map,
                                            "data": img_b64_str_map
                                        }
                                    },
                                    {
                                        "inline_data" : {
                                            "mime_type" : img_type_obs,
                                            "data": img_b64_obs
                                        }
                                    }
                                ]
                    })
            
            response = self.api_client.models.generate_content(
                model="gemini-2.0-flash", # not -exp
                contents=contents)
            ans = response.text
        print("Full correction prompt:\n", ans)

        match = re.search(r".*?Suggested intervention:\s*(\d+)", ans)

        if match:
            last_number = match.group(1)
            chosen_idx = int(last_number)
            return chosen_idx
        else:
            print("No number found")
            return -1
            


    def make_correction(self, idx):
        correction = self.corrections_idx[idx]
        actions = self.correction_to_action[correction]

        for lin, ang, freq in actions:
            for _ in range(freq):
                self.take_action(lin, ang)
                time.sleep(0.2)

    def llm_helper_runner(self):

        # Keep track of distance from goal 
        dist_check = 45 # if we haven't made progress (2 meters) in 120 seconds, that's a issue 
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
                # TAKE INTERVENTION 
                self.pause = True

                intervention = self.get_correction()
                if intervention != -1:
                    self.make_correction(intervention)

                self.replan(self.goal_pose_final, spacing = 50, keep_num = 3)
                    
                num_goals_before_replan = len(self.goal_stack) - 2
                latest_distances.clear()
                cur_dist = self.latest_goal_dist()
                latest_distances.append(cur_dist)
                last_update = time.time()
                last_check = time.time()
                
                self.pause = False
                print(f"New goal at {self.goal_stack[-1][0]:.8f} , {self.goal_stack[-1][1]:.8f} resume operation ")


    def model_runner(self):
        loop_time = 1 / 3
        start_time = time.time()    
          
        while True:
            cur_time = time.time()
            
            if cur_time - start_time > loop_time and not self.pause:
                linear, angular = self.compute_action()
                self.take_action(linear, angular)
                start_time = time.time()


    def run(self):
        data_thread = threading.Thread(target=self.get_robot_data_loop, daemon = True)
        main_thread = threading.Thread(target=self.llm_helper_runner, daemon = True)
        model_thread = threading.Thread(target=self.model_runner, daemon = True)

        data_thread.start()
        main_thread.start()
        model_thread.start()

        data_thread.join()
        main_thread.join()
        model_thread.join()


if __name__ == "__main__":
    cur_pos = np.array([37.873550 , -122.267617])
    campanile_pos = np.array([37.87210, -122.25780])

    points = np.vstack([cur_pos, campanile_pos])
    min_lat, min_long = np.min(points, axis = 0)
    max_lat, max_long = np.max(points, axis = 0)

    border = 0.001
    bounds = [[min_lat - border, min_long - border],
            [max_lat + border, max_long + border]]


    Actor(
        map_bounds = bounds,
        goal_pose = campanile_pos,
        obs_rate = 3,
        help_rate = 1,
        api_name = "gemini",
    ).run()
