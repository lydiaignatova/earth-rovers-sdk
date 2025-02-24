# generic imports
import time
import numpy as np
from PIL import Image
import os
import sys
import io
from dataclasses import dataclass
from typing import Optional
import tensorflow as tf
import time 
import atexit
import logging

import argparse

# custom imports
from agentlace.action import ActionClient
from agentlace.data.rlds_writer import RLDSWriter
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore

# data loading
from absl import app, flags, logging as absl_logging

import utils


IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])

MAX_TRAJ_LEN = 100 # 3 times a second * 30 seconds = 90 long
STEPS_TRY = 60
GOAL_DIST = STEPS_TRY // 2 # 4 - 10 # normal around this with 5 std 

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image to be between 0 and 1
    """
    return (tf.cast(image, tf.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD



class Recorder():

    def __init__(self, base_url, save_step, save_dir, max_time):

        self.action_client = ActionClient(
            server_ip="localhost",
            config=utils.make_action_config("frodobot")
        )

        self.tick_rate = 3
        self.max_time = max_time
        self.start_time = time.time()

        self.max_traj_len = save_step

        # set up save location 
        data_dir = save_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        existing_folders = [0] + [int(folder.split('.')[-1]) for folder in os.listdir(data_dir)]
        latest_version = max(existing_folders)

        self.version= f"0.0.{1 + latest_version}"
        self.datastore_path = f"{data_dir}/{self.version}"
        os.makedirs(self.datastore_path)

        
        # setting up writer
        self.traj_step = 0
        self.dtype = utils.observation_format_numpy()
        self.clear_obs()

        atexit.register(self.flush_data) 
        print("Datastore set up")

    def flush_data(self):
        if self.observations is None or len(self.observations) == 0:
            return

        first_img_time = int(self.observations[0][0]) # corresponds to timestamp_img
        file_prefix = f"{self.datastore_path}/{first_img_time}_{self.traj_step}"

        
        # save EVERYTHING ELSE
        obs_array = np.array(self.observations, dtype=self.dtype)
        np.savez(f"{file_prefix}_observations.npz", obs_array=obs_array)

        # save VIDEOS
        utils.write_video(self.imgs["front"], f"{file_prefix}_front.mp4", byte_string_frames = False, fps=30)
        utils.write_video(self.imgs["rear"], f"{file_prefix}_rear.mp4", byte_string_frames = False, fps=30)
        utils.write_video(self.imgs["map"], f"{file_prefix}_map.mp4", byte_string_frames = False, fps=30)

        print(f"Saved traj {self.traj_step}")

        # reset for next trajectory 
        self.traj_step += 1
        self.clear_obs()
        
    def clear_obs(self):
        self.observations = []
        self.imgs = {"front": [],
                     "rear": [],
                     "map": []}
        self.traj_len = 0

    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()

        while True:
            new_start_time = time.time()
            elapsed = new_start_time - start_time
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)
            start_time = time.time()

            self.tick()

    def save(self, obs):

        self.imgs["front"].append(utils.decode_from_base64(obs.pop("front_frame", None)))
        self.imgs["rear"].append(utils.decode_from_base64(obs.pop("rear_frame", None)))
        self.imgs["map"].append(utils.decode_from_base64(obs.pop("map_frame", None)))

        self.observations.append(utils.extract_ordered_values(obs, self.dtype))

    def tick(self): 
        obs = self.action_client.obs() 
        if obs is not None:
            self.save(obs)
            self.traj_len += 1

            if self.traj_len >= self.max_traj_len:
                self.flush_data()

            
        if self.max_time is not None:
            if time.time() - self.start_time > self.max_time:
                print(f"Killing recording after {time.time() - self.start_time} seconds")
                sys.exit()
            
if __name__ == "__main__":

    tf.get_logger().setLevel("WARNING")
    logging.basicConfig(level=logging.WARNING)
    absl_logging.set_verbosity("WARNING")

    parser = argparse.ArgumentParser(description='My Python Script')
    parser.add_argument('--data_save_dir', type=str, help='Where to save collected data')
    parser.add_argument('--save_step', type=int, default = 200, help='Frequency at which to split into trajectories')
    parser.add_argument('--max_time', type=int, help='How long to run for')
    parser.add_argument('--base_url', type=str, default="localhost",  help='What IP to connect to a robot action server on')
    args = parser.parse_args()

    Recorder(base_url= args.base_url,  
          save_dir = args.data_save_dir, 
          save_step = args.save_step,
          max_time = args.max_time,
          ).run() 