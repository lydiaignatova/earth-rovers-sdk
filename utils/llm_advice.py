import threading
import time
import requests

import numpy as np 
from collections import deque

import re
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


GOAL_THRES = 5 # 10 meters away - counts as reaching the goal! 

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

def resize_image_byte(img_b64_str, factor = 2): 
    img = byte_to_pil(img_b64_str)

    new_size = (img.width // factor, img.height // factor)
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


def points_between(start, end, num):
    x_points = np.linspace(start[0], end[0], num) 
    y_points = np.linspace(start[1], end[1], num) 

    points = [(x, y) for x, y in zip(x_points, y_points)]
    return points 


def dist_between_latlon(start, end):
     start_pos = utm.from_latlon(start[0], start[1])
     end_pos = utm.from_latlon(end[0], end[1])

     return np.linalg.norm(np.array(start_pos[:2]) - np.array(end_pos[:2]))

def get_tuples_list_floats(text):
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
        print("No path found in the text.")

class Map():

    def __init__(self, bounds, border=0.001):
        """ Bounds entered as [[min_lat, min_long], [max_lat, max_long]]"""

        self.marker_size = 9
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

    def plot_points(self, points, colors, lat_lon = True):
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

            self.ax.plot(lon, lat, marker="o", color=color, markersize=self.marker_size, transform=ccrs.PlateCarree())
            
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
        self.map = Map(map_bounds)
        self.map.plot_points([goal_pose], "blueviolet")
        self.no_pos = True
        self.map_lock = threading.Lock()

        # Goal Info 
        self.goal_pose_final = goal_pose
        self.goal_stack = deque()
        self.goal_stack.append(self.goal_pose_final)

        intermediate_goals = self.get_intermediate_goals(self.goal_pose_final)
        for i in range(2, -1, -1): # just get the closest 2 - recompute once we're there. 
            self.goal_stack.append(intermediate_goals[i])
        
        

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
            new_pos = (gpsdata["latitude"], gpsdata["longitude"], gpsdata["orientation"])
            self.position_deque.append(new_pos)

            # Add GPS data to map history if it isn't the same as before 
            if len(self.position_deque) < 2 or self.position_deque[-2] != new_pos:
                self.map_lock.acquire()
                try:
                    self.map.plot_points([new_pos[:2]], "cyan")
                finally:
                    self.map_lock.release()


    def latest_goal_dist(self):

        cur_pos = self.position_deque[-1] 
        cur_compass = -float(cur_pos[2])/180.0*3.141592 

        cur_utm = utm.from_latlon(cur_pos[0], cur_pos[1]) 
        goal_utm = utm.from_latlon(self.goal_stack[-1][0], self.goal_stack[-1][1]) 

        del_x, del_y = calculate_relative_position(cur_utm[0], cur_utm[1], 
                                                   goal_utm[0], goal_utm[1])
        
        return math.sqrt(del_x ** 2 + del_y ** 2)


    def get_intermediate_goals(self, goal_pos, spacing=50):

        # Figure out how many waypoints we want
        cur_pos = self.position_deque[-1]
        dist = dist_between_latlon(cur_pos, goal_pos)
        num = dist // spacing

        # Set up prompt
        goal_prompt = f"""I am a small wheeled root trying to navigate an environment. In this overhead map, my position is marked with a bright fuchsia dot, locations I have visited are marked with light blue dots, red dots indicate spots I have gotten stuck at or couldn't make progress from, and a purple dot represents my goal position. The map is split into a coordinate grid, with labels along the x and y axis. First, what is my approximate current position? What is the approximate goal position? Use precision up to 3 decimal spots, such as 2.568. Now, give me a path from my current position to the goal position using {num} intermediate waypoints. I am a wheeled robot, so going through buildings, which are marked in brown on the map, is impossible and causes me to get stuck. Always prioritize staying on white, gray, or red roads. This means you might sometimes need to move further away from the goal to ultimately make progress. Format the path like this [(x, y), (x, y), ..., (x, y)]. Don't include the start pose or the goal pose in the path."""

        # Set up map (plot cur pos, goal pos)
        self.map_lock.acquire()
        try:
            self.map.plot_points(cur_pos, "fuchsia")
            self.map.plot_points(goal_pos, "blueviolet")
            self.map.save_map("./goals.png")
        finally:
            self.map_lock.release()

        img_type_map = "image/png"
        img_b64_str_map = encode_image_to_base64("./goals.png")
        img_b64_str_map = resize_image_byte(img_b64_str_map)

        # Send Query
        self.api_start_time = time.time()
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
            input_images = [img_b64_str_map] 
            contents = [goal_prompt] +  input_images

            response = self.api_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents)
            ans = response.text

        goals = get_tuples_list_floats(ans) # extract out goal pos 

        print(f"response took {time.time() - self.api_start_time} seconds to generate \n")


        # Get goals from query 
        get_tuples_list_floats(response.text)
        # Plot + Refine goals 


        # add goals

        self.map.remove_points(1, "fuchsia")
        self.map.plot_points([gpsdata["longitude"]]), np.array([gpsdata["latitude"]], "fuchsia")

        
        start_pos = (5.000, 4.000)
        goal_pos = (11, 3.8)
        my_map.plot_points([start_pos], ["fuchsia"], False)
        # my_map.plot_points([goal_pos], ["blueviolet"], False)

        # my_map.plot_points([(4.900, 3.700)], ["blueviolet"], False)
        my_map.plot_points([campanile_pos], ["blueviolet"])
        # my_map.plot_points([(4.4, 3.0)], ["blueviolet"], False)

        my_map.save_map("griddy.png")

        # remove goals 
        

        return goals

    def runner(self):

        # Get starting distance from goal
        last_dist = self.latest_goal_dist()
        dist_check = 30 # if we haven't made progress (2 meters) in 30 seconds, that's a issue 
        last_check = time.time()

        while True:
            cur_dist = self.latest_goal_dist()
            
            # Check if we've reached the closest goal 
            if cur_dist < GOAL_THRES:
                
                self.goal_stack.pop()

                print("Goal while processing is", self.goal_stack[-1])

                intermediate_goals = self.get_intermediate_goals(self.goal_pose_final)
                self.goal_stack.pop()

                for i in range(2, -1, -1): # just get the closest 2 - recompute once we're there. 
                    self.goal_stack.append(intermediate_goals[i])

                print("New goal at", self.goal_stack[-1])

            
            # Check if we're stuck 
            if cur_dist + 2 >= last_dist:
                print("STUCK!")

                self.get_help_prompt 

            stack[-1] if stack


            # Check if we've made progress towards teh goal 


            # ask for help if it's been 5 seconds with on progress 


            # Get new intermediate goal 


            # Get latest map 
            img_type_map = "image/png"
            plt.savefig("./current_map.png", dpi=300, bbox_inches="tight")
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
            print("latest gps", gpsdata)
            cur_compass = -float(gpsdata[2])/180.0*3.141592 # don't reorient to have 0 as north 
            cur_utm = utm.from_latlon(gpsdata[0], gpsdata[1]) 
            goal_utm = utm.from_latlon()
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
        main_thread = threading.Thread(target=self.runner, daemon = False)

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
