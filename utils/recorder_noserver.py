# records GPS positions & image observations as videos

import time
import os
import sys
import utils 
import argparse
import atexit
import numpy as np 
import requests 


class Recorder():

    def __init__(self, save_dir, tick_rate):
        # Set up Variables
        self.tick_rate = tick_rate
        self.start_time = time.time()

        # Set up save location
        self.data_dir = save_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if len(os.listdir(self.data_dir)) > 0:
            print("Error: Directory is not empty.")
            sys.exit() 

        print("Save directory set up")
                                
        # Set up empty data
        self.clear_obs()
        atexit.register(self.flush_data) 

    def flush_data(self):
        # Save out GPS information 
        file_prefix = f"{self.data_dir}/{time.time()}"

        # save EVERYTHING ELSE
        gps_array = np.array(self.gps_data)
        np.savez(f"{self.data_dir}/gps.npz", gps=gps_array)

        # save VIDEOS
        utils.write_video(self.imgs["front"], f"{self.data_dir}/front.mp4", byte_string_frames = False, fps=15)
        utils.write_video(self.imgs["rear"], f"{self.data_dir}/rear.mp4", byte_string_frames = False, fps=15)

    def clear_obs(self):
        self.gps_data = []
        self.imgs = {"front": [],
                     "rear": [],
                     }
        self.traj_len = 0

    def get_data(self):
        # Get images
        screenshot = requests.get("http://127.0.0.1:8000/v2/screenshot")
        screenshot = screenshot.json() 

        self.imgs["front"].append(utils.decode_from_base64(screenshot["front_frame"]))
        self.imgs["rear"].append(utils.decode_from_base64(screenshot["rear_frame"]))

        # Get GPS data
        # gpsdata = requests.get("http://127.0.0.1:3000/last-data") # FROM MY SERVER
        gpsdata = requests.get("http://127.0.0.1:8000/data") # RAW FRODOBOTS DATA 
        gpsdata = gpsdata.json() 

        self.gps_data.append([gpsdata["latitude"], gpsdata["longitude"], gpsdata["orientation"]])

    def tick(self): 
        self.get_data()
        self.traj_len += 1

    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()    
        
        while True:
            cur_time = time.time()
            if cur_time - start_time > loop_time:
                self.tick()
                start_time = time.time()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Recorder')
    parser.add_argument('--data_save_dir', type=str, help='Where to save collected data')
    parser.add_argument('--tick_rate', type=int, default = 1, help='How many times per second to sample')
    args = parser.parse_args()

    Recorder(save_dir = args.data_save_dir, 
          tick_rate = args.tick_rate,
          ).run() 
