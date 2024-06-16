import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .shelf import RAFT
from .interpolation import interpolate
from dot.utils.io import read_config
from dot.utils.torch import get_grid, get_sobel_kernel

import matplotlib.pyplot as plt

import random
import numpy as np
import cv2

from dot.utils.io import create_folder, write_video, write_frame
from dot.utils.plot import to_rgb, plot_points
import os.path as osp

def vis_gaussian_weighting(tensor1, tensor2, i, j):
    """ Creates figure of the weights for debugging purposes """
    tensor1 = tensor1.squeeze(0)
    tensor2 = tensor2.squeeze(0)

    channel_1 = tensor1[:, :, 0]
    channel_2 = tensor2[:, :, 0]

    channel_1_cpu = channel_1.cpu().numpy()
    channel_2_cpu = channel_2.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1= axes[0].imshow(channel_1_cpu, cmap='gray')
    axes[0].set_title('Regular Gaussian')
    plt.colorbar(im1, ax=axes[0], orientation='vertical', label='Value')

    im2 = axes[1].imshow(channel_2_cpu, cmap='gray')
    axes[1].set_title('Fast Gaussian')
    plt.colorbar(im2, ax=axes[1], orientation='vertical', label='Value')

    plt.savefig(f'gaussian_weighting_{i}_{j}.png')
    plt.close(fig)

def apply_gaussian_weights(alpha, cotracker_predictions, sigma, use_alpha=False):
    """Applies Gaussian Filter to coordinates of the cotracker tracks with the given sigma.
    This function uses torch operations optimized for GPU to efficiently calculate weighted averages
    of alpha channel values based on spatial distances from cotracker predictions.

    Parameters:
        alpha (torch.Tensor): A 3D tensor of shape (1, H, W) where H is the height, and W is the width of the input feature map.
        cotracker_predictions (torch.Tensor): A 3D tensor of shape (N, 3) where N is the number of tracks.
                              Each prediction contains the x coordinate, y coordinate, and a visibility value.
        sigma (float): Standard deviation of the Gaussian distribution used for weighing.
        use_alpha (bool): If True, multiplies the result by the input alpha tensor.

    Returns:
        torch.Tensor: A tensor of the same height and width as 'alpha', but with two channels because it needs to match the flow field,
                      replicated across the last dimension. Shape (1, H, W, 2)
    """
    y_coords, x_coords = torch.meshgrid(torch.arange(alpha.size(1)), torch.arange(alpha.size(2)))
    x_coords = x_coords.float().to(device="cuda:0").unsqueeze(0)
    y_coords = y_coords.float().to(device="cuda:0").unsqueeze(0)

    cotracker_predictions = cotracker_predictions.squeeze(0).unsqueeze(1).unsqueeze(2)
    
    distances_squared = (x_coords - cotracker_predictions[:,:,:,0])**2 + (y_coords - cotracker_predictions[:,:,:,1])**2
    
    weights = torch.exp(-distances_squared / (2 * sigma**2))
    weights *= cotracker_predictions[:,:,:,2]
    weighted_alpha = torch.sum(weights, dim=0)
    if use_alpha: weighted_alpha *= alpha
    weighted_alpha /= torch.max(weighted_alpha)

    return weighted_alpha.unsqueeze(-1).repeat(1, 1, 1, 2)



class OpticalFlow(nn.Module):
    def __init__(self, height, width, config, load_path):
        super().__init__()
        model_args = read_config(config)
        model_dict = {"raft": RAFT}
        self.model = model_dict[model_args.name](model_args)
        self.name = model_args.name
        if load_path is not None:
            device = next(self.model.parameters()).device
            self.model.load_state_dict(torch.load(load_path, map_location=device))
        coarse_height, coarse_width = height // model_args.patch_size, width // model_args.patch_size
        self.register_buffer("coarse_grid", get_grid(coarse_height, coarse_width))
        self.refined_flow = dict() # aka. self.target, refined_flow[i][j] (i, j are index for consecutive frames)
        self.refined_weight = dict() # aka. self.weight
        self.refined_flow_inac = dict()
        self.refined_weight_inac = dict()
        self.export_epe = False

    def forward(self, data, mode, **kwargs):
        if mode == "flow_between_frames":
            return self.get_flow_between_frames(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
        
    def reset_inac(self, ii, jj):
        # remove the ii&jj from inac
        for idx in range(len(ii)):
            i = ii[idx]
            j = jj[idx]
            self.refined_flow_inac[i].pop(j)
            self.refined_weight_inac[i].pop(j)
        torch.cuda.empty_cache()
        return
    
    def rm_flows(self, ii, jj, store=False):
        # print(f'optical_flow, rm_flows: removing frame {ii} to {jj}, with store={store}')
        
        if store:
            # move the stored refined_flow and weight to ..._inac
            for idx in range(len(ii)):
                i = ii[idx]
                j = jj[idx]
                if i not in self.refined_flow_inac.keys():
                    self.refined_flow_inac[i] = dict()
                    self.refined_weight_inac[i] = dict()
                
                self.refined_flow_inac[i][j] = self.refined_flow[i][j].to('cpu')
                self.refined_weight_inac[i][j] = self.refined_weight[i][j].to('cpu')
        # delete the refined_flow and weight
        if store:
            for idx in range(len(ii)):
                i = ii[idx]
                j = jj[idx]
                self.refined_flow[i].pop(j)
                self.refined_weight[i].pop(j)
        torch.cuda.empty_cache()
        return
    
    def get_flow_between_frames(self, track, video, ii, jj):
        """
        Input: 
            - track: track from CoTracker
            - video: torch.Size([1000, 3, 384, 512])
            - ii: torch.Size([60])
            - jj: torch.Size([60])
        Output:
            - target: torch.Size([1, 60, 48, 64, 2])
        """
        l = len(ii)
        
        for idx in range(l):
            i = ii[idx]
            j = jj[idx]
            if i not in self.refined_flow.keys(): #and i not in self.refined_weight_inac.keys():
                self.refined_flow[i] = dict()
                self.refined_weight[i] = dict()
            else:
                if i in self.refined_flow.keys() and j in self.refined_flow[i].keys():
                    continue
                elif i in self.refined_flow_inac.keys() and j in self.refined_flow_inac[i].keys():
                    continue
                
            #print(f'optical flow: getting refined flow between frame {i} to {j}')
            src_points = track[:, i].to('cuda')
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            
            coarse_flow, coarse_alpha, _ = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                    version="torch3d")
            flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                    tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                    src_feats=data["src_feats"] if "src_feats" in data else None,
                                    tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                    coarse_flow=coarse_flow,
                                    coarse_alpha=coarse_alpha,
                                    is_train=False,
                                    slam_refinement=True)
            
            H, W = 512, 512
            gaussian_src_points = src_points.clone().detach()
            gaussian_src_points[...,0] = gaussian_src_points[...,0]*(W-1)
            gaussian_src_points[...,1] = gaussian_src_points[...,1]*(H-1)
            weighted_alpha = apply_gaussian_weights(alpha, gaussian_src_points, (H+W)*0.01)
            self.refined_flow[i][j] = flow[0].to('cpu')
            self.refined_weight[i][j] = weighted_alpha[0].to('cpu')
        torch.cuda.empty_cache()

    def get_flow_magnitude(self, pairs, track, video, coord0):
        for (i, j) in pairs:
            src_points = track[:, i].to('cuda')
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            coarse_flow, coarse_alpha, _ = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                    version="torch3d")
            flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                    tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                    src_feats=data["src_feats"] if "src_feats" in data else None,
                                    tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                    coarse_flow=coarse_flow,
                                    coarse_alpha=coarse_alpha,
                                    is_train=False,
                                    slam_refinement=True)
            scale_x, scale_y = 64/128, 48/128
            flow += coord0
            resized_flow = cv2.resize(flow[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]

        for (i, j) in pairs:
            src_points = track[:, i].to('cuda')
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            coarse_flow, coarse_alpha, _ = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                    version="torch3d")
            flow_up, alpha_up = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                src_feats=data["src_feats"] if "src_feats" in data else None,
                                tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                coarse_flow=coarse_flow,
                                coarse_alpha=coarse_alpha,
                                is_train=False,
                                slam_refinement=False)
            scale_x, scale_y = 64/512, 48/512
            resized_flow = cv2.resize(flow_up[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            
    def save_flows_for_epe(self, pairs, track, video):
        for (i, j) in pairs:
            src_points = track[:, i].to('cuda')
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            coarse_flow, coarse_alpha, _ = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                    version="torch3d")
            flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                    tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                    src_feats=data["src_feats"] if "src_feats" in data else None,
                                    tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                    coarse_flow=coarse_flow,
                                    coarse_alpha=coarse_alpha,
                                    is_train=False,
                                    slam_refinement=True)
            scale_x, scale_y = 640/128, 480/128
            resized_flow = cv2.resize(flow[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            np.save(f'./flow_grid_epe/resized_flow_{i}_{j}.npy', resized_flow)

            flow_up, alpha_up = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                src_feats=data["src_feats"] if "src_feats" in data else None,
                                tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                coarse_flow=coarse_flow,
                                coarse_alpha=coarse_alpha,
                                is_train=False,
                                slam_refinement=False)
            scale_x, scale_y = 640/512, 480/512
            resized_flow = cv2.resize(flow_up[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            np.save(f'./flow_grid_epe/upscale_resized_flow_{i}_{j}.npy', resized_flow)




