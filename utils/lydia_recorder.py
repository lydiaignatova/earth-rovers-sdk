# TFDS RECORDER!!!! that's the plan <3 


# just one long ass trajectory i guess for now


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

    def __init__(self, base_url, save_dir, max_time):

        self.action_client = ActionClient(
            server_ip="localhost",
            config=utils.make_action_config("frodobot")
        )

        self.tick_rate = 3
        self.max_time = max_time
        self.start_time = time.time()

        self.max_traj_len = 200

        # set up save location 
        data_dir = save_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        existing_folders = [0] + [int(folder.split('.')[-1]) for folder in os.listdir(data_dir)]
        latest_version = max(existing_folders)

        self.version= f"0.0.{1 + latest_version}"
        self.datastore_path = f"{data_dir}/{self.version}"
        os.makedirs(self.datastore_path)

        
        # setting up rlds writer
        self.image_size = (64, 64)
        data_spec = utils.record_data_format("frodobot")
    
        self.writer = RLDSWriter(
            dataset_name="test",
            data_spec = data_spec,
            data_directory = self.datastore_path,
            version = self.version,
            max_episodes_per_file = 100,
        )

        atexit.register(self.writer.close) 

        self.data_store = EpisodicTFDataStore(
            capacity=1000,
            data_spec= data_spec,
            rlds_logger = self.writer
        )
        print("Datastore set up")
            
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()

        self.first = True
        self.last = False
        self.terminal = False

        self.just_crashed = False
        self.traj_len = 0
        self.curr_goal = None 

        while True:
            new_start_time = time.time()
            elapsed = new_start_time - start_time
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)
            start_time = time.time()

            self.tick()

    def int_image(self, img):
        return np.asarray((img * IMAGENET_STD + IMAGENET_MEAN) * 255, dtype = np.uint8)

    def save(self, obs):
        # convert everything to float32! 
        obs["front_frame"] = tf.convert_to_tensor(obs["front_frame"]) 

        for k in ["battery", "signal_level", "orientation", "lamp"]:
            # print(f"{k} has type{type(obs[k])}")
            obs[k] = tf.cast(obs[k], dtype=tf.int32)
                  
        for k in ["timestamp_data", "timestamp_img", "speed", "gps_signal",
                  "latitude", "longitude", "vibration", "accels",
                  "gyros", "mags", "rpms", "last_action_linear", "last_action_angular"]:
            # print(f"{k} has type{type(obs[k])}")
            obs[k] = tf.cast(obs[k], dtype=tf.float32)   
   

        formatted_obs = {
            "observation": obs,
            "action": tf.concat([obs["last_action_linear"], obs["last_action_angular"]], axis = 0),
            "is_first": self.first, 
            "is_last": self.last, 
            "is_terminal": self.terminal, 
        }

        if self.first:
            self.first = False

        if self.last:
            print("End of trajectory")
            self.first = True
            self.last = False

        self.data_store.insert(formatted_obs)

    def tick(self): 
        obs = self.action_client.obs() 
        if obs is not None:
            self.save(obs)
            self.traj_len += 1

            if self.traj_len >= self.max_traj_len:
                self.traj_len = 0
                self.last = True

            
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
    parser.add_argument('--max_time', type=int, help='How long to run for')
    parser.add_argument('--base_url', type=str, default="localhost",  help='What IP to connect to a robot action server on')
    args = parser.parse_args()

    Recorder(base_url= args.base_url,  
          save_dir = args.data_save_dir, 
          max_time = args.max_time,
          ).run() 