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
# from agentlace.action import ActionServer, ActionConfig
import imageio
import numpy as np 
import math 
import utm 
import re


from openai import OpenAI

from google import genai

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


#####################
### POSITION DATA ###
#####################

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


def points_between(start, end, num):
    x_points = np.linspace(start[0], end[0], num) 
    y_points = np.linspace(start[1], end[1], num) 

    points = [(x, y) for x, y in zip(x_points, y_points)]
    return points 


def dist_between_latlon(start, end):
     start_pos = utm.from_latlon(start[0], start[1])
     end_pos = utm.from_latlon(end[0], end[1])

     return np.linalg.norm(np.array(start_pos[:2]) - np.array(end_pos[:2]))

##################
### IMAGE DATA ###
##################

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        img_b64_bytes = base64.b64encode(image_file.read())
        img_b64_str = img_b64_bytes.decode("utf-8")
    return img_b64_str

def resize_image_byte(img_b64_str, factor = 2): 
    img = byte_to_pil(img_b64_str)

    new_size = (int(img.width / factor), int(img.height / factor))
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

################
### LLM DATA ###
################


def get_tuples_list_floats(text):
    # Get rid of comments 
    text = re.sub(r'#.*', '', text)  # Removes inline and standalone comments

    # Extract the path using a regular expression
    match = re.search(r'\[\s*(\([^)]*\),?\s*)+\]', text)  # Matches the entire list of tuples

    if match:
        path_string = match.group(0)  # Get the matched string

        # Clean up the string and split it into tuples
        path_string = path_string.replace(" ", "")  # Remove spaces to enable proper splitting
        path_string = path_string.replace("[", "")  # Remove the leading bracket
        path_string = path_string.replace("]", "")  # Remove the ending bracket
        tuples_string = path_string.split("),")    # Split into strings representing tuples
        tuples_string = [t.replace(")", "") for t in tuples_string] # Clean out last bracket in each tuple

        # Convert the string tuples to a list of float tuples
        path = []
        for t in tuples_string:
            coords = t.replace("(", "").split(",") # Clean out brackets and split the numbers in the tuple
            try:
                x, y = float(coords[0]), float(coords[1])
                path.append((x, y))
            except (ValueError, IndexError) as e:
                print(f"Skipping invalid tuple format: {t}. Error: {e}") # Handles the missing bracket in each split.

        return path 
    else:
        print(f"No path found in the text. {text}")


########### 
### MAP ###
###########

class Map():

    def __init__(self, bounds, border=0.001):
        """ Bounds entered as [[min_lat, min_long], [max_lat, max_long]]"""

        self.bounds = bounds 
        self.min_lat, self.min_long = self.bounds[0]
        self.max_lat, self.max_long = self.bounds[1]

        tiler = cimgt.OSM()
        crs = tiler.crs

        self.fig, self.ax = plt.subplots(
            figsize=(12, 10),
            subplot_kw={"projection": crs}
        )

        self.ax.set_extent([bounds[0][1], bounds[1][1] , 
                    bounds[0][0], bounds[1][0]], 
                    crs=ccrs.PlateCarree())
        self.ax.add_image(tiler, 18)  # Add OpenStreetMap tiles; zoom level lower values = less zoomed-in
        
        self.rows = self.cols = None 

    def save_map(self, path):
        plt.savefig(path, dpi=300, bbox_inches="tight")

    def visualize(self):
        plt.show()

    def add_grid(self, rows, cols):
        if self.rows is not None or self.cols is not None:
            raise ValueError("Grid already added to map, cannot add another one.")
        
        self.gridlines = []
        self.gridlabels = []
        self.rows, self.cols = rows, cols

        # Generate lat/lon grid points
        latitudes = np.linspace(self.min_lat, self.max_lat, self.rows + 1)
        longitudes = np.linspace(self.min_long, self.max_long, self.cols + 1)

        # Plot horizontal grid lines and add row labels
        for i, lat in enumerate(latitudes):
            line = self.ax.plot([self.min_long, self.max_long], [lat, lat], 
                    color="black", linewidth=0.5, linestyle="--", 
                    transform=ccrs.PlateCarree())

            row_label = f"{i}" 
            label = self.ax.text(self.min_long - 0.0003, lat, row_label, 
                    fontsize=10, ha="right", va="center", color="black",
                    transform=ccrs.PlateCarree())
            
            self.gridlines.append(line)
            self.gridlabels.append(label)

        # Plot vertical grid lines and add column labels
        for j, lon in enumerate(longitudes):
            line = self.ax.plot([lon, lon], [self.min_lat, self.max_lat], 
                    color="black", linewidth=0.5, linestyle="--", 
                    transform=ccrs.PlateCarree())

            col_label = f"{j}" 
            label = self.ax.text(lon, self.min_lat - 0.0003, col_label, 
                    fontsize=10, ha="center", va="top", color="black",
                    transform=ccrs.PlateCarree())
            
            self.gridlines.append(line)
            self.gridlabels.append(label)

    def remove_grid(self):
        if self.rows is None and self.cols is None:
            raise ValueError("Grid has not been added to map, cannot remove.")
        
        for line in self.gridlines:
            line[0].remove()

        for label in self.gridlabels:
            label.remove()

        self.rows = self.cols =  None
        self.gridlines = self.gridlabels = None

    def plot_points(self, points, colors, lat_lon = True, size = 9):
        if isinstance(colors, list) and not isinstance(colors, str):
            if len(points) != len(colors):
                raise ValueError(f"Must have lengths of lists for plotting equal, have {len(points)} points but {len(colors)} colors")
        else:
            color = colors 

        for i in range(len(points)):
            lat, lon = points[i]

            if not lat_lon:
                lat, lon = self.coords_to_lat_long(lat, lon)

            if isinstance(colors, list) and not isinstance(colors, str):
                color = colors[i]

            self.ax.plot(lon, lat, marker="o", color=color, markersize=size, transform=ccrs.PlateCarree())
            
    def remove_points(self, num, color):
        count = 0
        for line in reversed(self.ax.lines):  
            if line.get_marker() == "o" and line.get_color() == color:
                line.remove()
                count += 1
                if count >= num:  # Stop after removing 'num' points
                    break  

    def coords_to_lat_long(self, x, y):
        if self.rows is None or self.cols is None:
            raise ValueError("Cannot compute coordinates from lat long without a grid")
        
        long  = self.min_long + x * (self.max_long - self.min_long) / self.cols
        lat = self.min_lat + y * (self.max_lat - self.min_lat) / self.rows

        return lat, long 
    
    def lat_long_to_coords(self, lat, long):
        if self.rows is None or self.cols is None:
            raise ValueError("Cannot compute lat long from coordinates without a grid")
        
        x = (long - self.min_long) / ((self.max_long - self.min_long) / self.cols)
        y = (lat - self.min_lat) / ((self.max_lat - self.min_lat) / self.rows)

        return x, y 


###############
### SDK USE ###
###############


def data_request(url):
    url = f"{url}/data"
    response = requests.get(url)
    response = response.text
    return response

def observation_format_numpy():
    return [
        ('timestamp_img', 'f4'),    # Float32
        ('timestamp_data', 'f4'),
        
        ('battery', 'i4'),          # Int32
        ('signal_level', 'i4'),
        ('orientation', 'i4'),
        ('lamp', 'i4'),
        ('speed', 'f4'),            # Float32
        ('gps_signal', 'f4'),
        ('latitude', 'f4'),         # Float32 for precision with coordinates
        ('longitude', 'f4'),
        ('vibration', 'f4'),

        ('accels', 'f4', (6, 4)),   # 6x4 Float32 matrix
        ('gyros', 'f4', (5, 4)),    # 5x4 Float32 matrix
        ('mags', 'f4', (1, 4)),     # 1x4 Float32 matrix
        ('rpms', 'f4', (5, 5)),     # 5x5 Float32 matrix

        ('last_action_linear', 'f4', (3,)),   # Float32 array of size 3
        ('last_action_angular', 'f4', (3,)),  # Float32 array of size 3
    ]

# Function to extract values in the correct order
def extract_ordered_values(observation, dtype):
    return tuple(observation[key[0]] for key in dtype)

def observation_format(observation_key_type):
    if observation_key_type == "frodobot":
        return    {
            "front_frame": tf.TensorSpec((), tf.string, name="front_frame"),
            "rear_frame": tf.TensorSpec((), tf.string, name="rear_frame"),
            "map_frame": tf.TensorSpec((), tf.string, name="map_frame"),

            "timestamp_img": tf.TensorSpec((), tf.float32, name="timestamp_img"),
            "timestamp_data": tf.TensorSpec((), tf.float32, name="timestamp_data"),

            "battery": tf.TensorSpec((), tf.int32, name="battery"),
            "signal_level": tf.TensorSpec((), tf.int32, name="signal_level"),
            "orientation": tf.TensorSpec((), tf.int32, name="orientation"),
            "lamp": tf.TensorSpec((), tf.int32, name="lamp"),
            "speed": tf.TensorSpec((), tf.float32, name="speed"),
            "gps_signal": tf.TensorSpec((), tf.float32, name="gps_signal"),
            "latitude": tf.TensorSpec((), tf.float32, name="latitude"),  # for precision with coordinates
            "longitude": tf.TensorSpec((), tf.float32, name="longitude"), # for precision with coordinates
            "vibration": tf.TensorSpec((), tf.float32, name="vibration"),
            # "timestamp": tf.TensorSpec((), tf.float32, name="timestamp"), # 64-bit for timestamp precision
            "accels": tf.TensorSpec((6, 4), tf.float32, name="accels"),   # 6x4 matrix for accelerometer data
            "gyros": tf.TensorSpec((5, 4), tf.float32, name="gyros"),     # 5x4 matrix for gyroscope data
            "mags": tf.TensorSpec((1, 4), tf.float32, name="mags"),       # 1x4 matrix for magnetometer data
            "rpms": tf.TensorSpec((5, 5), tf.float32, name="rpms"),       # 5x5 matrix for RPM data

            # "action_state_source": tf.TensorSpec((), tf.string, name="action_state_source"),
            "last_action_linear": tf.TensorSpec((3,), tf.float32, name="last_action_linear"),
            "last_action_angular": tf.TensorSpec((3,), tf.float32, name="last_action_angular"),
        }
    else: 
        raise ValueError(f"Unknown observation config type {observation_key_type}")



def record_data_format(observation_key_type):
    """ FOR RLDS (need is_first, is_last, is_terminal)"""
    return {
        "observation": {
            **observation_format(observation_key_type),
        },
        "action": tf.TensorSpec((6,), tf.float32, name="action"),
        "is_first": tf.TensorSpec((), tf.bool, name="is_first"),
        "is_last": tf.TensorSpec((), tf.bool, name="is_last"),
        "is_terminal": tf.TensorSpec((), tf.bool, name="is_terminal"),
    }

def make_action_config(action_config_type):
    if action_config_type == "frodobot":
        return ActionConfig(
            port_number=1235,
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

def write_video(frames, save_path, byte_string_frames = False, fps=30):

    writer = imageio.get_writer(save_path, fps=fps)

    for frame in frames:
        if frame is not None:
            if byte_string_frames:
                frame = decode_from_base64(frame)
            frame = np.array(frame)
            writer.append_data(frame)
        
    writer.close()

## PROCESSING DATA
def decode_from_base64(base64_string):
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    return image