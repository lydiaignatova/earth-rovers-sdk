## copying things in from the jupyter notebook i want to unclog
%load_ext autoreload
%autoreload 2

from agentlace.action import ActionServer, ActionConfig, ActionClient
import tensorflow as tf
import numpy as np
import utils
import matplotlib.pyplot as plt
import time
import os

import imageio

### NEW DATA FORMAT
base_dir = f"/home/lydia/data/frodobot/supervised"
traj_dir = f"{base_dir}/fri25-20-24-9am"
all_files = os.listdir(traj_dir)
all_files = sorted(all_files)

npzs = [f for f in all_files if f[-4:] == ".npz"]
images = {
    "front": [f for f in all_files if f[-5:] == "t.mp4"],
    "rear": [f for f in all_files if f[-5:] == "r.mp4"],
    "map": [f for f in all_files if f[-5:] == "p.mp4"],
}

# process NPZs
arrays = []
for npz in npzs:
    data = np.load(os.path.join(traj_dir, npz))
    arrays.append(data["obs_array"])

concat_arr = np.concatenate(arrays, axis = 0)
print(f"have {concat_arr.size} data points with data type {concat_arr.dtype}")

# process movie frames

# takes about a second per 200 frame video... 
def get_video_frames(video_path):
    frames = []
    reader = imageio.get_reader(video_path)
    total_frames = reader.count_frames()
    for frame in reader:
        frames.append(np.array(frame))
    reader.close()
    return np.array(frames)

video_file = images["front"][0]
ans = get_video_frames(os.path.join(traj_dir, video_file))

# 60 seconds for 39 200 frame videos ...
front_frames_all = [get_video_frames(os.path.join(traj_dir, vid)) for vid in images["front"]]

plt.imshow(front_frames_all[35][158])

hi = np.load("/home/lydia/data/frodobot/supervised/0.0.31/1729571465_0_observations.npz")
hi["obs_array"]


### FRODO DATA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

base_dir = "/home/lydia/data/frodobots/theirs/ride_20699_17c99e_20240310191305"
ride_num = 20699

control_data = pd.read_csv(f"{base_dir}/control_data_{ride_num}.csv")
front_camera_timestamps = pd.read_csv(f"{base_dir}/front_camera_timestamps_{ride_num}.csv")
rear_camera_timestamps = pd.read_csv(f"{base_dir}/rear_camera_timestamps_{ride_num}.csv")
gps_data = pd.read_csv(f"{base_dir}/gps_data_{ride_num}.csv")
imu_data = pd.read_csv(f"{base_dir}/imu_data_{ride_num}.csv")
speaker_audio_timestamps = pd.read_csv(f"{base_dir}/speaker_audio_timestamps_{ride_num}.csv")

plt.plot(gps_data["latitude"], gps_data["longitude"])
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.axis("equal")
plt.title("FrodoBot Position")
plt.show()


### DROID SLAM

import numpy as np
import matplotlib.pyplot as plt

recon_dir =  "/home/lydia/repos/DROID-SLAM/reconstructions/data/frodorecon"
disps = np.load(f"{recon_dir}/disps.npy")
intrinsics = np.load(f"{recon_dir}/intrinsics.npy")
poses = np.load(f"{recon_dir}/poses.npy")
tstamps = np.load(f"{recon_dir}/tstamps.npy")
imgs = np.load(f"{recon_dir}/images.npy")

imgs.shape

i = 15
cmapped = np.array([imgs[i][1], imgs[i][2], imgs[i][0]])
plt.imshow(cmapped.transpose(1, 2, 0))
plt.show()

colors = plt.cm.viridis(np.linspace(0,1,poses.shape[0])) #.jet is rainbow :D
plt.scatter(poses[:, 0], poses[:, 1], color=colors[:, :3])
plt.axis("equal")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].

ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], color=colors[:, :3])

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# Example 3D points that approximately lie on a plane
points = poses[:, :3]

# 1. Perform PCA to find the normal of the plane
pca = PCA(n_components=3)
pca.fit(points)
normal_vector = pca.components_[-1]  # The normal corresponds to the smallest eigenvalue

# 2. Compute the rotation needed to align the normal vector with the z-axis
# The z-axis is [0, 0, 1], so we want to rotate the normal vector to align with [0, 0, 1]
z_axis = np.array([0, 0, 1])

# Compute the axis of rotation (cross product of normal and z-axis)
rotation_axis = np.cross(normal_vector, z_axis)

# Compute the angle of rotation (arccos of the dot product between the vectors)
rotation_angle = np.arccos(np.dot(normal_vector, z_axis) / (np.linalg.norm(normal_vector) * np.linalg.norm(z_axis)))

# Create a rotation object using scipy's Rotation class
rotation = R.from_rotvec(rotation_axis * rotation_angle)

# 3. Apply the rotation to the points
rotated_points = rotation.apply(points)

# 4. Plot the original and rotated points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121, projection='3d')
ax.set_title("Original 3D Points")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors, label='Original Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.axis("equal")

# Plot the rotated points
ax_rot = fig.add_subplot(122, projection='3d')
ax_rot.set_title("Rotated 3D Points (Plane-aligned)")
ax_rot.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], color=colors, label='Rotated Points')
ax_rot.set_xlabel('X')
ax_rot.set_ylabel('Y')
ax_rot.set_zlabel('Z')
ax_rot.axis("equal")

plt.show()

plt.scatter(rotated_points[:, 0], rotated_points[:, 1], color=colors)
plt.axis("equal")
plt.show()

plt.plot(np.arange(len(tstamps)), tstamps)
plt.show()
