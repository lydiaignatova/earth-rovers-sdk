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
    def __init__(self, goal_gps, goal_utm, goal_compass):

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

        # Save goals 
        self.goal_gps = goal_gps
        self.goal_utm = goal_utm
        self.goal_compass = goal_compass 
        self.goal_id = 0 # start with the first goal 

        # Processing information
        self.context_queue = deque(maxlen = self.context_size + 1)
        self.img_size = (96, 96)
        self.tick_rate = 3


    def get_img_data(self):
        response = requests.get("http://127.0.0.1:8000/v2/front")
        response = response.json() 
           
        img_PIL = decode_from_base64(response["front_frame"])
        img_PIL = img_PIL.resize(self.img_size)
        img_PIL = img_PIL.convert('RGB')

        self.context_queue.append(img_PIL)

        return img_PIL 
        

    def get_gps_data(self):
        gpsdata = requests.get("http://127.0.0.1:3000/last-data")
        gpsdata = gpsdata.json() 

        robotdata = requests.get("http://127.0.0.1:8000/data")
        robotdata = robotdata.json()   

        # cur_gps = (gpsdata["latitude"], gpsdata["longitude"])
        cur_gps = (robotdata["latitude"], robotdata["longitude"])
        cur_utm = utm.from_latlon(cur_gps[0], cur_gps[1]) 
        cur_compass = -float(robotdata["orientation"])/180.0*3.141592 

        return cur_gps, cur_utm, cur_compass

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

        img_PIL = self.get_img_data()
        cur_gps, cur_utm, cur_compass = self.get_gps_data()

        if len(self.context_queue) < self.context_size + 1:
            return 0, 0 # no action while not enough context 
        
        else:
            input_imgs = transform_images_exaug(list(self.context_queue))
            input_imgs = input_imgs.to(device)

            delta_x, delta_y = calculate_relative_position(cur_utm[0], cur_utm[1], self.goal_utm[self.goal_id][0], self.goal_utm[self.goal_id][1])
            relative_x, relative_y = rotate_to_local_frame(delta_x, delta_y, cur_compass)
            
            # Check if reached goal 
            if np.sqrt(relative_x**2 + relative_y**2) < THRES_UPDATE:
                print("\n\n REACHED GOAL")
                if self.goal_id != len(self.goal_compass)-1:
                    self.goal_id += 1   
                    print("MOVING ONTO NEXT GOAL")

            # If goal is too far (past THRES_DIST), project it closer 
            if np.sqrt(relative_x**2 + relative_y**2) > THRES_DIST:
                relative_x = relative_x/np.sqrt(relative_x**2 + relative_y**2)*THRES_DIST
                relative_y = relative_y/np.sqrt(relative_x**2 + relative_y**2)*THRES_DIST   
            
            goal_pose = np.array([relative_y/METRIC_WAYPOINT_SPACING, -relative_x/METRIC_WAYPOINT_SPACING, np.cos(self.goal_compass[self.goal_id]-cur_compass), np.sin(self.goal_compass[self.goal_id]-cur_compass)])
            goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
            
            print("\nCurrent position", cur_gps, "orientation", cur_compass)
            print("Goal position", self.goal_gps[self.goal_id])
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

    def tick(self):
        linear, angular = self.compute_action()
        self.take_action(linear, angular)


    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()    
          
        while True:
            cur_time = time.time()
            
            if cur_time - start_time > loop_time:
                self.tick()
                start_time = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='My Python Script')
    parser.add_argument('--blah', type=str, default="no", help='Where to save collected data')
    args = parser.parse_args()

    # latlon = [[37.871574761893605, -122.26096307116137], [37.872195093371765, -122.26180548297141], [37.87195822250516, -122.26280161953142], [37.871894357599786, -122.26420674854921]]
    # goal_compass = [-10/180*3.14, -100/180*3.14, -100/180*3.14, -120/180*3.14]  

    # going by li ka shing down to browns 
    # latlon = [[37.87240406613363, -122.26528394060068], [37.87203323041331, -122.26439699617309], [37.872536783338006, -122.26368887853576], [37.87297641312939, -122.26446738205651]]
    # goal_compass = [-160/180*3.14, -80/180*3.14, 0, 90/180*3.14]

    # Circle ccw
    latlon = [[37.87326966603712, -122.26717737694783], 
              [37.87314060204639, -122.26835765883582], 
              [37.87382221869118, -122.26847006663466], 
              [37.87389078333003, -122.26757591368921]]
    goal_compass = [-180/180*3.14, 90/180*3.14, 0, 90/180*3.14]

    goal_utm = []
    for i in range(len(latlon)):
        goal_utm.append(utm.from_latlon(latlon[i][0], latlon[i][1]))
    
    print("goal", goal_utm[len(latlon)-1], goal_compass[len(latlon)-1])

    Actor(
          goal_gps = latlon,
          goal_utm = goal_utm,
          goal_compass = goal_compass,
          ).run() 
    
