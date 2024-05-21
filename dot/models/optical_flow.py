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
    tensor1 = tensor1.squeeze(0)
    tensor2 = tensor2.squeeze(0)

    channel_1 = tensor1[:, :, 0]
    channel_2 = tensor2[:, :, 0]

    channel_1_cpu = channel_1.cpu().numpy()
    channel_2_cpu = channel_2.cpu().numpy()

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first channel
    im1= axes[0].imshow(channel_1_cpu, cmap='gray')
    axes[0].set_title('Regular Gaussian')
    plt.colorbar(im1, ax=axes[0], orientation='vertical', label='Value')

    # Plot the second channel
    im2 = axes[1].imshow(channel_2_cpu, cmap='gray')
    axes[1].set_title('Fast Gaussian')
    plt.colorbar(im2, ax=axes[1], orientation='vertical', label='Value')

    # Save the figure
    plt.savefig(f'gaussian_weighting_{i}_{j}.png')
    plt.close(fig)

def old_apply_gaussian_weights(alpha, cotracker_predictions, sigma):
    y_coords, x_coords = torch.meshgrid(torch.arange(alpha.size(1)), torch.arange(alpha.size(2)))
    x_coords = x_coords.float().to(device="cuda:0", dtype=torch.long)
    y_coords = y_coords.float().to(device="cuda:0", dtype=torch.long)
    
    cotracker_predictions = cotracker_predictions.squeeze(0)
    
    weighted_alpha = torch.zeros_like(alpha)
    
    for confident_point in cotracker_predictions:
        if confident_point[2]: # if track visible at that point TODO
            distances_squared = (x_coords - confident_point[0])**2 + (y_coords - confident_point[1])**2
            weights = torch.exp(-distances_squared / (2 * sigma**2))
            weighted_alpha += weights
    
    weighted_alpha *= alpha
    weighted_alpha /= torch.max(weighted_alpha)
    weighted_alpha = weighted_alpha.unsqueeze(-1)
    
    return weighted_alpha.repeat(1, 1, 1, 2) # duplicate every value along a new dimension to achieve torch.Size([1, 512, 512, 2])

def apply_gaussian_weights(alpha, cotracker_predictions, sigma):
    y_coords, x_coords = torch.meshgrid(torch.arange(alpha.size(1)), torch.arange(alpha.size(2)))
    x_coords = x_coords.float().to(device="cuda:0").unsqueeze(0)
    y_coords = y_coords.float().to(device="cuda:0").unsqueeze(0)

    cotracker_predictions = cotracker_predictions.squeeze(0).unsqueeze(1).unsqueeze(2)
    
    distances_squared = (x_coords - cotracker_predictions[:,:,:,0])**2 + (y_coords - cotracker_predictions[:,:,:,1])**2
    
    weights = torch.exp(-distances_squared / (2 * sigma**2))
    weights *= cotracker_predictions[:,:,:,2]
    weighted_alpha = torch.sum(weights, dim=0) # * alpha
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
        if mode == "flow_with_tracks_init":
            return self.get_flow_with_tracks_init(data, **kwargs)
        elif mode == "motion_boundaries":
            return self.get_motion_boundaries(data, **kwargs)
        elif mode == "feats":
            return self.get_feats(data, **kwargs)
        elif mode == "tracks_for_queries":
            return self.get_tracks_for_queries(data, **kwargs)
        elif mode == "tracks_from_first_to_every_other_frame":
            return self.get_tracks_from_first_to_every_other_frame(data, **kwargs)
        elif mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        elif mode == "flow_between_frames":
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
        # print(f'optical_flow: current flows: {self.refined_flow.keys()}')
        if store:
            # move the stored refined_flow and weight to ..._inac
            for idx in range(len(ii)):
                i = ii[idx]
                j = jj[idx]
                if i not in self.refined_flow_inac.keys():
                    self.refined_flow_inac[i] = dict()
                    self.refined_weight_inac[i] = dict()
                # print('keys for {i}:', self.refined_flow[i].keys())
                self.refined_flow_inac[i][j] = self.refined_flow[i][j]#.to('cpu')
                self.refined_weight_inac[i][j] = self.refined_weight[i][j]#.to('cpu')
        # delete the refined_flow and weight
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
        # T, C, h, w = video.shape
        # H, W = 512, 512
        # # reshape video 
        # # TODO: reshape it when inputing instead of doing it every time here
        # if h != H or w != W:
        #     video = F.interpolate(video, size=(H, W), mode="bilinear")
        #     video = video.reshape(T, C, H, W)[None]
        # else:
        #     video = video[None] # add dimension (batch)

        l = len(ii)
        # target, weight = [], []
        for idx in range(l):
            i = ii[idx]
            j = jj[idx]
            if i not in self.refined_flow.keys():# and i not in self.refined_weight_inac.keys():
                self.refined_flow[i] = dict()
                self.refined_weight[i] = dict()
            else:
                if i in self.refined_flow.keys() and j in self.refined_flow[i].keys():
                    continue
                elif i in self.refined_flow_inac.keys() and j in self.refined_flow_inac[i].keys():
                    # move from inac to ac
                    self.refined_flow[i][j] = self.refined_flow_inac[i][j]
                    self.refined_weight[i][j] = self.refined_weight_inac[i][j]
                    self.refined_flow_inac[i].pop(j)
                    self.refined_weight_inac[i].pop(j)
                    continue
                
            # print(f'optical flow: getting refined flow between frame {i} to {j}')
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
            # pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
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
            
            # if random.random() < 0.07:
            #     print(f'optical_flow: visualizing flow {i}-{j}//////////////')
            #     write_frame(to_rgb(flow, "flow")[0], osp.join('/home/wangwen/DOT-SLAM/flow_visualize', f"pred_flow_{i}_{j}.png"))

            # if [int(i), int(j)] in [[15, 19], [17, 15], [21, 15], [26, 24], [79, 80], [94, 96], [100, 104], [114, 117], [122, 118], [126, 122], [127, 128], [135, 133]]:
            # # if random.random() < 0.03:
            #     scale_x, scale_y = 640/128, 480/128
            #     print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            #     resized_flow = cv2.resize(flow[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     resized_flow = resized_flow * [scale_x, scale_y]
            #     print('optical_flow: reshaped flow.shape', resized_flow.shape)
            #     np.save(f'./flow_grid_epe/resized_flow_{i}_{j}.npy', resized_flow)

            #     flow_up, alpha_up = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
            #                         tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
            #                         src_feats=data["src_feats"] if "src_feats" in data else None,
            #                         tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
            #                         coarse_flow=coarse_flow,
            #                         coarse_alpha=coarse_alpha,
            #                         is_train=False,
            #                         slam_refinement=False)
            #     scale_x, scale_y = 640/512, 480/512
            #     print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            #     resized_flow = cv2.resize(flow_up[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            #     resized_flow = resized_flow * [scale_x, scale_y]
            #     print('optical_flow: reshaped flow.shape', resized_flow.shape)
            #     np.save(f'./flow_grid_epe/upscale_resized_flow_{i}_{j}.npy', resized_flow)
            # # TODO: make this dynamic (only works for 512, 512 right now)
            H, W = 512, 512
            # weighted_alpha = old_apply_gaussian_weights(alpha, data["src_points"], (H+W)*0.01) # 0.05 -> divide by two for height H and width W, and divide by 10 for the weigthing 
            # bring the coordinates back to image dimension:
            gaussian_src_points = src_points.clone().detach()
            gaussian_src_points[...,0] = gaussian_src_points[...,0]*(W-1)
            gaussian_src_points[...,1] = gaussian_src_points[...,1]*(H-1)
            weighted_alpha = apply_gaussian_weights(alpha, gaussian_src_points, (H+W)*0.01)
            self.refined_flow[i][j] = flow[0]#.to('cpu')
            self.refined_weight[i][j] = weighted_alpha[0]#.to('cpu')

            
            # target.append(flow[0])
            # weight.append(weighted_alpha[0])
        # target = torch.stack(target, dim=0)[None]
        # weight = torch.stack(weight, dim=0)[None]
        # if len(track[0]) > 80:
        #     self.get_flow_magnitude(track=track, video=video)
        # if (not self.export_epe) and len(track[0]) > 140:
        #     self.save_flows_for_epe(track=track, video=video)
        #     self.export_epe = True
        torch.cuda.empty_cache()
        # return target, weight
    
    def get_flow_magnitude(self, track, video, coord0):
        return
        #print('************* get_flow_magnitude *************')
        pairs = [(3, 6), (3, 10), (3, 14), (6, 3), (6, 10), (6, 14), (6, 18), (10, 3), (10, 6), (10, 14), (10, 18), (10, 21), (14, 3), (14, 6), (14, 10), (14, 18), (14, 21), (14, 24), (18, 6), (18, 10), (18, 14), (18, 21), (18, 24), (18, 27), (21, 10), (21, 14), (21, 18), (21, 24), (21, 27), (21, 30), (24, 14), (24, 18), (24, 21), (24, 27), (24, 30), (24, 34), (27, 18), (27, 21)]
        for (i, j) in pairs:
            # print(f'get_flow_magnitude: optical flow: getting refined flow between frame {i} to {j}')
            src_points = track[:, i].to('cuda')
            # src_frame =  video[:, i]
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            # tgt_frame =  video[:, j]
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            # pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
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
            # print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            resized_flow = cv2.resize(flow[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            # print('optical_flow: reshaped flow.shape', resized_flow.shape)
            #print(f'flow magnitude (128*128 -> 48*64): {i}-{j}: {torch.mean(torch.norm(torch.permute(torch.tensor(resized_flow)[None], (0, 3, 1, 2)), dim=1), dim=(1, 2))}')

        for (i, j) in pairs:
            # print(f'get_flow_magnitude: optical flow: getting refined flow between frame {i} to {j}')
            src_points = track[:, i].to('cuda')
            # src_frame =  video[:, i]
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            # tgt_frame =  video[:, j]
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            # pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
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
            # print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            resized_flow = cv2.resize(flow_up[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            # print('optical_flow: reshaped flow.shape', resized_flow.shape)
            #print(f'flow magnitude (512*512 -> 48*64): {i}-{j}: {torch.mean(torch.norm(torch.permute(torch.tensor(resized_flow)[None], (0, 3, 1, 2)), dim=1), dim=(1, 2))}')
            
        #print('************* all computed *************')
        assert False
            
    def save_flows_for_epe(self, track, video):
        return 
        #print('************* save_flows_for_epe *************')
        for (i, j) in [[15, 19], [17, 15], [21, 15], [26, 24], [79, 80], [94, 96], [100, 104], [114, 117], [122, 118], [126, 122], [127, 128], [135, 133]]:
            #print(f'save_flows_for_epe: optical flow: getting refined flow between frame {i} to {j}')
            src_points = track[:, i].to('cuda')
            # src_frame =  video[:, i]
            src_frame =  video[i][None].cuda()
            tgt_points = track[:, j].to('cuda')
            # tgt_frame =  video[:, j]
            tgt_frame =  video[j][None].cuda()

            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            # pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
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
            #print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            resized_flow = cv2.resize(flow[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            #print('optical_flow: reshaped flow.shape', resized_flow.shape)
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
            #print(f'optical_flow, saved flow[{i}][{j}].shape:', flow.shape)
            resized_flow = cv2.resize(flow_up[0].clone().cpu().numpy(), None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            resized_flow = resized_flow * [scale_x, scale_y]
            #print('optical_flow: reshaped flow.shape', resized_flow.shape)
            np.save(f'./flow_grid_epe/upscale_resized_flow_{i}_{j}.npy', resized_flow)
        #print('************* all saved *************')
        assert False

    def get_motion_boundaries(self, data, boundaries_size=1, boundaries_dilation=4, boundaries_thresh=0.025, **kwargs):
        eps = 1e-12
        src_frame, tgt_frame = data["src_frame"], data["tgt_frame"]
        K = boundaries_size * 2 + 1
        D = boundaries_dilation
        B, _, H, W = src_frame.shape
        reflect = torch.nn.ReflectionPad2d(K // 2)
        sobel_kernel = get_sobel_kernel(K).to(src_frame.device)
        flow, _ = self.model(src_frame, tgt_frame)
        norm_flow = torch.stack([flow[..., 0] / (W - 1), flow[..., 1] / (H - 1)], dim=-1)
        norm_flow = norm_flow.permute(0, 3, 1, 2).reshape(-1, 1, H, W)
        boundaries = F.conv2d(reflect(norm_flow), sobel_kernel)
        boundaries = ((boundaries ** 2).sum(dim=1, keepdim=True) + eps).sqrt()
        boundaries = boundaries.view(-1, 2, H, W).mean(dim=1, keepdim=True)
        if boundaries_dilation > 1:
            boundaries = torch.nn.functional.max_pool2d(boundaries, kernel_size=D * 2, stride=1, padding=D)
            boundaries = boundaries[:, :, -H:, -W:]
        boundaries = boundaries[:, 0]
        boundaries = boundaries - boundaries.reshape(B, -1).min(dim=1)[0].reshape(B, 1, 1)
        boundaries = boundaries / boundaries.reshape(B, -1).max(dim=1)[0].reshape(B, 1, 1)
        boundaries = boundaries > boundaries_thresh
        return {"motion_boundaries": boundaries, "flow": flow}

    def get_feats(self, data, **kwargs):
        video = data["video"]
        feats = []
        for step in tqdm(range(video.size(1)), desc="Extract feats for frame", leave=False):
            feats.append(self.model.encode(video[:, step]))
        feats = torch.stack(feats, dim=1)
        return {"feats": feats}

    def get_flow_with_tracks_init(self, data, is_train=False, interpolation_version="torch3d", alpha_thresh=0.8, **kwargs):
        coarse_flow, coarse_alpha, _ = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                version=interpolation_version)
        flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                 tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                 src_feats=data["src_feats"] if "src_feats" in data else None,
                                 tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                 coarse_flow=coarse_flow,
                                 coarse_alpha=coarse_alpha,
                                 is_train=is_train)
        if not is_train:
            alpha = (alpha > alpha_thresh).float()
        return {"flow": flow, "alpha": alpha, "coarse_flow": coarse_flow, "coarse_alpha": coarse_alpha}

    def get_tracks_for_queries(self, data, **kwargs):
        raise NotImplementedError




