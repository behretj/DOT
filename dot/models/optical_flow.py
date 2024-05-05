import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from .shelf import RAFT
from .interpolation import interpolate
from dot.utils.io import read_config
from dot.utils.torch import get_grid, get_sobel_kernel


def old_apply_gaussian_weights(alpha, cotracker_predictions, sigma):
    x_coords, y_coords = torch.meshgrid(torch.arange(alpha.size(0)), torch.arange(alpha.size(1)))
    x_coords = x_coords.float()
    y_coords = y_coords.float()
    
    weighted_alpha = torch.zeros_like(alpha, dtype=torch.float64)
    
    for confident_point in cotracker_predictions:
        distances_squared = (x_coords - confident_point[0])**2 + (y_coords - confident_point[1])**2
        weights = torch.exp(-distances_squared / (2 * sigma**2))
        weighted_alpha += weights
    
    # Normalization:
    weighted_alpha /= torch.max(weighted_alpha)

    weighted_alpha *= alpha
    
    return weighted_alpha

def apply_gaussian_weights(alpha, cotracker_predictions, sigma):
    x_coords, y_coords = torch.meshgrid(torch.arange(alpha.size(0)), torch.arange(alpha.size(1)))
    x_coords = x_coords.float()
    y_coords = y_coords.float()

    x_coords = x_coords.unsqueeze(-1)
    y_coords = y_coords.unsqueeze(-1)

    cotracker_predictions = cotracker_predictions.unsqueeze(0).unsqueeze(0)

    distances_squared = (x_coords - cotracker_predictions[..., 0])**2 + (y_coords - cotracker_predictions[..., 1])**2
    
    weights = torch.exp(-distances_squared / (2 * sigma**2))
    weighted_alpha = weights.sum(dim=-1)
    weighted_alpha *= alpha    
    weighted_alpha /= torch.max(weighted_alpha)
    
    return weighted_alpha


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
        T, C, h, w = video.shape
        H, W = 512, 512
        # reshape video 
        # TODO: reshape it when inputing instead of doing it every time here
        if h != H or w != W:
            video = F.interpolate(video, size=(H, W), mode="bilinear")
            video = video.reshape(T, C, H, W)[None]
        else:
            video = video[None] # add dimension (batch)

        l = len(ii)
        target, weight = [], []
        for idx in range(l):
            i = ii[idx]
            j = jj[idx]
            src_points = track[:, i]
            src_frame =  video[:, i]
            tgt_points = track[:, j]
            tgt_frame =  video[:, j]
            data = {
                "src_frame": src_frame,
                "tgt_frame": tgt_frame,
                "src_points": src_points,
                "tgt_points": tgt_points
            }
            # pred = self.optical_flow_refiner(data, mode="flow_with_tracks_init", **kwargs)
            coarse_flow, coarse_alpha = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
                                                    version="torch3d")
            ## TODO: I need alpha here, so reverse the change slam_refinement=True part
            flow, alpha = self.model(src_frame=data["src_frame"] if "src_feats" not in data else None,
                                    tgt_frame=data["tgt_frame"] if "tgt_feats" not in data else None,
                                    src_feats=data["src_feats"] if "src_feats" in data else None,
                                    tgt_feats=data["tgt_feats"] if "tgt_feats" in data else None,
                                    coarse_flow=coarse_flow,
                                    coarse_alpha=coarse_alpha,
                                    is_train=False)
            # TODO: check dimensions of data["src_points"]
            weighted_alpha = apply_gaussian_weights(alpha, data["src_points"], (H+W)*0.05) # 0.05 -> divide by two for height H and width W, and divide by 10 for the weigthing 
            weight.append(weighted_alpha)
            target.append(flow)
        target = torch.stack(target, dim=0)[None]
        return target, weight
            

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
        coarse_flow, coarse_alpha = interpolate(data["src_points"], data["tgt_points"], self.coarse_grid,
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




