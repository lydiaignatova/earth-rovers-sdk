import numpy as np
from agentlace.action import ActionServer, ActionConfig
import tensorflow as tf
import threading
import requests
import json
import time 
import utils 


from typing import Any, Optional, Tuple, Set, Callable


class FrodoServer():

    def __init__(self, base_url, raw_img):

        self.base_url = base_url

        self.frame_rate = 3
        self.data_rate = 5
        
        # Empty Obs
        self._latest_obs = {
            "front_frame": np.array(b"", dtype=bytes),
            "rear_frame": np.array(b"", dtype=bytes),
            "map_frame": np.array(b"", dtype=bytes),

            # Timestamp
            "timestamp_img": np.zeros((), dtype=np.float32),
            "timestamp_data": np.zeros((), dtype=np.float32),

            # Other sensor data
            "battery": np.zeros((), dtype=np.int32),
            "signal_level": np.zeros((), dtype=np.int32),
            "orientation": np.zeros((), dtype=np.int32),
            "lamp": np.zeros((), dtype=np.int32),
            "speed": np.zeros((), dtype=np.float32),
            "gps_signal": np.zeros((), dtype=np.float32),
            "latitude": np.zeros((), dtype=np.float32),
            "longitude": np.zeros((), dtype=np.float32),
            "vibration": np.zeros((), dtype=np.float32),

            # Acceleration, gyroscope, magnetometer, and RPM data
            "accels": np.zeros((6, 4), dtype=np.float32),
            "gyros": np.zeros((5, 4), dtype=np.float32),
            "mags": np.zeros((1, 4), dtype=np.float32),
            "rpms": np.zeros((5, 5), dtype=np.float32),

            # "action_state_source": np.zeros((), dtype=str),
            "last_action_linear": np.zeros((3,), dtype=np.float32),
            "last_action_angular": np.zeros((3,), dtype=np.float32),
        }
        print("Observation type set up")

        self.action_server = ActionServer(
            config=utils.make_action_config("frodobot"),
            obs_callback=self.agentlace_obs_callback,
            act_callback=self.agentlace_act_callback,
        )

        # Start running
        print("action server started")

        # Start reading in data
        self.start()


    # get the current observation
    def agentlace_obs_callback(self, keys: Set[str]):
        return {k: self._latest_obs[k] for k in keys}
    
    def agentlace_act_callback(self, key: str, payload: Any):
        if key == "action_vw":
            result = self.receive_vw_action_callback(payload)
        else:
            result = {"running": False, "reason": f"Unknown key {key}"}
        return result
    
    def receive_vw_action_callback(self, command: np.ndarray):
        linear, angular = command[0], command[1]
        url = f"{self.base_url}/control"
        data = json.dumps({"command": {"linear": linear, "angular": angular}})
        
        response = requests.post(url, data=data)
        response = response.json()

        self._latest_obs["last_action_linear"] = np.array([linear, 0.0, 0.0])
        self._latest_obs["last_action_angular"] = np.array([0.0, 0.0, angular])
        
        if response["message"] == 'Command sent successfully':
            return True 
        else:
            return False


    def data_request(self, lock):
        url = f"{self.base_url}/data"
        response = requests.get(url)
        if response is None:
            return False
        
        response = response.json()
        if response is None:
            return False
        
        response["timestamp_data"] = response.pop("timestamp", None)

        # response["timestamp_data"] = tf.strings.to_number(response.pop("timestamp", None), out_type=tf.float32) 
        
        with lock:
            for key in response.keys():
                self._latest_obs[key] = response[key]
            
        return True
    
    def image_request(self, lock):
        url = f"{self.base_url}/screenshot"
        response = requests.get(url) 
        if response is None:
            return False
        
        response = response.json()
        if response is None:
            return False
        
        response["timestamp_img"] = response.pop("timestamp", None)

        with lock:
            for key in response.keys():
                self._latest_obs[key] = response[key]
        
        return True
    
    def image_loop(self, lock):
        while True:
            self.image_request(lock)
            time.sleep(1/self.frame_rate)
    
    def data_loop(self, lock):
        while True:
            self.data_request(lock)
            time.sleep(1/self.data_rate)
    

    def start(self):
        # data requests 
        lock = threading.Lock()
        image_thread = threading.Thread(target=self.image_loop, args=(lock,))
        data_thread = threading.Thread(target=self.data_loop, args=(lock,))
        image_thread.start()
        data_thread.start()

        # action server start
        self.action_server.start(threaded=True)

if __name__ == "__main__":

    import logging
    logging.basicConfig(level=logging.WARNING)

    # parser = argparse.ArgumentParser(description='Robot Action Server')
    # parser.add_argument('--raw_img', action='store_true')
    # args = parser.parse_args()

    node = FrodoServer("http://localhost:8000", raw_img = False)
    node.start()


    

        


