    

DIR_loc = "/home/lydia/Documents/work/rail/repos/Learning-to-Drive-Anywhere-via-MBRA"

import sys
sys.path.insert(0, DIR_loc + '/train')

from vint_train.models.il.il import IL_dist, IL_gps

import yaml
import torch 
import argparse 

import numpy as np 
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


MODEL_WEIGHTS_PATH = DIR_loc + "/policy/MBL_based_gps"
MODEL_CONFIG_PATH = DIR_loc + "/train/config"

# def transform_images_exaug(pil_imgs: List[PILImage.Image]) -> torch.Tensor:
#     """Transforms a list of PIL image to a torch tensor."""
#     transform_type = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                                     0.229, 0.224, 0.225]),
#         ]
#     )
#     if type(pil_imgs) != list:
#         pil_imgs = [pil_imgs]
#     transf_imgs = []
#     for pil_img in pil_imgs:
#         w, h = pil_img.size

#         transf_img = transform_type(pil_img)
#         transf_img = torch.unsqueeze(transf_img, 0)
#         transf_imgs.append(transf_img)
#     return torch.cat(transf_imgs, dim=1)



def main(args):

    model_config_path = MODEL_CONFIG_PATH + "/" + "frodobot_dist_IL2_gps.yaml"    
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)

    context_size = config["context_size"]


    model = IL_gps(
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
    
    print("Model Loaded ")
    
    model_path = MODEL_WEIGHTS_PATH + "/" + "latest.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    print("Model weights loaded ")

    # SAMPLE RUN 
    goal_pose = np.array([100.0, 0.0, 1.0, 0.0])
    goal_pose_torch = torch.from_numpy(goal_pose).unsqueeze(0).float().to(device)
    
    # metric_waypoint_spacing = 0.25
    # print("relative pose", goal_pose[0]*metric_waypoint_spacing, goal_pose[1]*metric_waypoint_spacing, goal_pose[2], goal_pose[3])
    
    obs_images = torch.zeros(1, (context_size  + 1)* 3, 200, 200)
    obs_images = obs_images.to(device)

    with torch.no_grad():  
        waypoints = model(obs_images, goal_pose_torch)   

    print(f"waypoints generated {waypoints}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Loading Model Test')
    parser.add_argument('--data_save_dir', type=str, help='Where to save collected data')
    parser.add_argument('--save_step', type=int, default = 200, help='Frequency at which to split into trajectories')
    parser.add_argument('--max_time', type=int, help='How long to run for')
    parser.add_argument('--base_url', type=str, default="localhost",  help='What IP to connect to a robot action server on')
    args = parser.parse_args()

    main(args)

