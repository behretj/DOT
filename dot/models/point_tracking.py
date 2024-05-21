from tqdm import tqdm
import torch
from torch import nn
import time
import cv2
import numpy as np

from .optical_flow import OpticalFlow
from .shelf import CoTracker, CoTracker2, Tapir, CoTracker2Online
from dot.utils.io import read_config
from dot.utils.torch import sample_points, sample_mask_points, get_grid

import matplotlib.pyplot as plt
from scipy import ndimage
import math









def vis_harris(Ncorners, src_frame):
    Ncorners = Ncorners.cpu()
    image = src_frame.squeeze().permute(1, 2, 0).cpu().numpy()

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.scatter(Ncorners[:, 0], Ncorners[:, 1], c='red', s=40, marker='o')  # Plot corners as red points

    # Remove axis ticks for better visualization
    ax.set_xticks([])
    ax.set_yticks([])

    # Save the plot as an image
    fig.savefig('corners_visualization.png')
    plt.close(fig)  # Close the figure to free up memory

class PointTracker(nn.Module):
    def __init__(self,  height, width, tracker_config, tracker_path, estimator_config, estimator_path, isOnline=False):
        super().__init__()
        model_args = read_config(tracker_config)

        sampling_inititization_functions  = {
            'harris' : self.init_harris,
            'grid' : self.init_grid
            }
        self.init_sampl_func = sampling_inititization_functions['grid']

        if isOnline:
            self.OnlineCoTracker_initialized = False
            self.modelOnline = CoTracker2Online(model_args)
            if tracker_path is not None:
                device = next(self.modelOnline.parameters()).device
                self.modelOnline.load_state_dict(torch.load(tracker_path, map_location=device), strict=False)
        else:
            model_dict = {
                "cotracker": CoTracker,
                "cotracker2": CoTracker2,
                "tapir": Tapir,
                "bootstapir": Tapir
            }
            self.name = model_args.name
            self.model = model_dict[model_args.name](model_args)
            if tracker_path is not None:
                device = next(self.model.parameters()).device
                self.model.load_state_dict(torch.load(tracker_path, map_location=device), strict=False)
        self.optical_flow_estimator = OpticalFlow(height, width, estimator_config, estimator_path)

    def forward(self, data, mode, **kwargs):
        if mode == "tracks_at_motion_boundaries":
            return self.get_tracks_at_motion_boundaries(data, **kwargs)
        elif mode == "tracks_at_motion_boundaries_online_droid":
            return self.get_tracks_at_motion_boundaries_online_droid(data, **kwargs)
        elif mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    #def harris_n_corner_detection(self, src_frame_tensor, n_keypoints): 
    #    if n_keypoints <= 0:
    #        return torch.empty((0, 2)).to('cuda')
    #    src_frame_tensor = src_frame_tensor.to('cpu')
    #    r, g, b = src_frame_tensor[:, 0, :, :], src_frame_tensor[:, 1, :, :], src_frame_tensor[:, 2, :, :]
    #    grayscale_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b  # according to the human eye may not be best here
    #    grayscale_tensor = torch.permute(grayscale_tensor, (1, 2, 0))
    #    grayscale_numpy = grayscale_tensor.numpy()
    #    dst = cv2.cornerHarris(grayscale_numpy, 2, 3, 0.04)
    #    # get the N strongest corners indexes
    #    flattened_dst_strongest_corner_indexes = np.argpartition(dst.flatten(), -n_keypoints)[-n_keypoints:]
    #    Ncorners = torch.stack(torch.unravel_index(torch.from_numpy(flattened_dst_strongest_corner_indexes), dst.shape),
    #                           dim=1)
    #    return Ncorners.to('cuda')

    def dst_harris_computation(self, src_frame_tensor):
        src_frame_tensor = src_frame_tensor.to('cpu')
        r, g, b = src_frame_tensor[:, 0, :, :], src_frame_tensor[:, 1, :, :], src_frame_tensor[:, 2, :, :]
        grayscale_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b  # according to the human eye may not be best here
        grayscale_tensor = torch.permute(grayscale_tensor, (1, 2, 0))
        grayscale_numpy = grayscale_tensor.numpy()
        dst = cv2.cornerHarris(grayscale_numpy, 2, 3, 0.04)
        return dst

    def grid_n_corner_detection(self, src_frame_tensor, n_keypoints):
        if n_keypoints <= 0:
            return torch.empty((0,2))
        #print('src_frame_tensor.shape', src_frame_tensor.shape)
        b, c, height, width = src_frame_tensor.shape
        num = int(math.sqrt(n_keypoints))
        # h_step, w_step = height // int(1 + math.sqrt(n_keypoints)), width // int(1 + math.sqrt(n_keypoints))
        # grid_points = []
        # for y in range(h_step, height - h_step, h_step):
        #     for x in range(w_step, width - w_step, w_step):
        #         grid_points.append([x, y])
        # # print('grid_points', grid_points)
        # # print('n_keypoints', n_keypoints)
        # # print(f'getting {n_keypoints} grids from image shape {src_frame_tensor.shape}:', len(grid_points))
        # x, y = 3, 3
        # while n_keypoints > len(grid_points):
        #     grid_points.append([x, y])
        #     x += w_step
        #     y += h_step

        grid_a = np.linspace(1, width - 1, num)
        grid_b = np.linspace(1, height - 1, num)
        grid_x, grid_y = np.meshgrid(grid_a, grid_b)
        grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.int16)
        x, y = 2, 3
        while n_keypoints > grid_points.shape[0]:
            grid_points = np.vstack((grid_points, np.array([x, y], dtype=np.int16)))
            x += 3
            y += 3
        # print(f'n_keypoints: {n_keypoints}, grid_points.shape: {grid_points.shape}')
        return torch.tensor(grid_points)

    def init_harris(self, data, num_tracks_max=8192, sim_tracks=2000,
                    sample_mode="first", init_queries_first_frame=torch.empty((0, 2)).to('cuda'),
                    nbr_grid_cell_width=20, nbr_grid_cell_height=20,
                    **kwargs):

        start = time.time()
        video_chunck = data["video_chunk"]

        B, T, _, H, W = video_chunck.shape
        assert T >= 1  # require at least two frame to get motion boundaries (the motion boundaries are computed between frame 0 and 1

        backward_tracking = True
        flip = False

        if flip:
            video_chunck = video_chunck.flip(dims=[1])

        src_points = []
        src_step = 0
        src_frame = video_chunck[:, src_step]

       
        # add the new keypoint to replace the keypoints we lost

        nbr_per_cell = sim_tracks // (nbr_grid_cell_width * nbr_grid_cell_height)

        difference = sim_tracks % (nbr_grid_cell_width * nbr_grid_cell_height)

        cell_width = src_frame.shape[2] // nbr_grid_cell_width
        cell_height = src_frame.shape[3] // nbr_grid_cell_height

        # nbr_new_keypoints_per_cell = np.full([[nbr_per_cell]*nbr_grid_cell_height]* nbr_grid_cell_width))#sim_tracks-init_queries_first_frame.shape[0]
        nbr_new_keypoints_per_cell = np.full((nbr_grid_cell_width, nbr_grid_cell_height), nbr_per_cell)

        for i in range(init_queries_first_frame.shape[0]):
            cell_w_index = int(min(nbr_grid_cell_width - 1, max(0, init_queries_first_frame[i][
                0]) // cell_width))  # we clip the point to be in frame (in one the grid cell even if they just went out)
            cell_h_index = int(min(nbr_grid_cell_height - 1, max(0, init_queries_first_frame[i][
                1]) // cell_height))  # we clip the point to be in frame (in one the grid cell even if they just went out)
            nbr_new_keypoints_per_cell[cell_w_index, cell_h_index] -= 1

        harris_dist = self.dst_harris_computation(src_frame)

        nbr_point_resampled = 0

        queries_2d_coords = init_queries_first_frame.to('cpu')
        for cell_w_index in range(nbr_grid_cell_width):
            for cell_h_index in range(nbr_grid_cell_height):
                local_dist = harris_dist[cell_w_index * cell_width:(cell_w_index + 1) * cell_width,
                             cell_h_index * cell_height:(cell_h_index + 1) * cell_height]
                nbr_keypoint_to_resample_in_cell = max(0, nbr_new_keypoints_per_cell[cell_w_index, cell_h_index])
                if difference > 0:
                    difference -= 1
                    nbr_keypoint_to_resample_in_cell += 1
                if nbr_keypoint_to_resample_in_cell <= 0: continue
                nbr_point_resampled += nbr_keypoint_to_resample_in_cell
                # get the N strongest corners indexes
                flattened_dst_strongest_corner_indexes = np.argpartition(local_dist.flatten(),
                                                                         -nbr_keypoint_to_resample_in_cell)[
                                                         -nbr_keypoint_to_resample_in_cell:]
                Ncorners = torch.stack(
                    torch.unravel_index(torch.from_numpy(flattened_dst_strongest_corner_indexes), local_dist.shape),
                    dim=1)
                Ncorners[:, 0] += cell_w_index * cell_width
                Ncorners[:, 1] += cell_h_index * cell_height
                queries_2d_coords = torch.cat((queries_2d_coords, Ncorners), dim=0)

        # print("Nbr of points resampled / kept / new number of keypoints: ", nbr_point_resampled,
            #   init_queries_first_frame.shape[0], queries_2d_coords.shape[0], flush=True)

        # print("points kept during resampling : ",init_queries_first_frame)

        # add the prior = the keypoint still visible from the last tracks

        # vis_harris(queries_2d_coords, src_frame)

        src_steps_tensor = torch.full((queries_2d_coords.shape[0], 1), src_step)
        src_corners = torch.cat((src_steps_tensor, queries_2d_coords), dim=1)  # coordonate contain src_frame_index
        src_corners = torch.stack([src_corners], dim=0)
        src_points.append(src_corners)

        # src_points[0].shape torch.Size([1, 64, 3])
        src_points = torch.cat(src_points, dim=1)

        # src_points = torch.Size([1, 64, 3]) #3 = (frame=0, height_y width_x)

        _, _ = self.modelOnline(video_chunck.to('cuda'), src_points.to('cuda'), is_first_step=True)
        self.OnlineCoTracker_initialized = True




    def init_grid(self, data, num_tracks_max=1600, sim_tracks=1600,
                                                        sample_mode="first", init_queries_first_frame=torch.empty((0, 2)).to('cuda'),
                                                        **kwargs):

            N, S = num_tracks_max, sim_tracks  # num_tracks, sim_tracks
            start = time.time()
            video_chunck = data["video_chunk"]

            B, T, _, H, W = video_chunck.shape
            assert T>=1 #require at least two frame to get motion boundaries (the motion boundaries are computed between frame 0 and 1
            

            backward_tracking = True
            flip = False
           

            if flip:
                video_chunck = video_chunck.flip(dims=[1])

            src_points = []
            src_step = 0
            nbr_samples = S
            src_frame = video_chunck[:, src_step]

           
            #add the new keypoint to replace the keypoints we lost
            nbr_new_keypoint = nbr_samples-init_queries_first_frame.shape[0]

            src_frame.shape[2]//2
            center_point = (src_frame.shape[2]//2, src_frame.shape[3]//2)
            


            to_resample = [nbr_samples//4]*4

            difference_left = nbr_samples-sum(to_resample)
            # print(difference_left)
            # print(to_resample)


            for i in range(difference_left):
                to_resample[i%4] +=1
            # print(to_resample)

            for i in range(init_queries_first_frame.shape[0]):
                if init_queries_first_frame[i][0]<=center_point[0]:
                    if init_queries_first_frame[i][1]<=center_point[1]:
                        to_resample[0] -= 1
                    else:
                        to_resample[1] -= 1
                else: 
                    if init_queries_first_frame[i][1]<=center_point[1]:
                        to_resample[2] -= 1
                    else:
                        to_resample[3] -= 1

            # print(to_resample)
            for i in range(2*len(center_point)):
                if to_resample[i%4] < 0:
                    to_resample[(i+1)%4] += to_resample[i%4]
                    to_resample[i%4] = 0

            
            Ncorners = self.grid_n_corner_detection(src_frame[:,:,:center_point[0],:center_point[1]], max(0,to_resample[0])) #TODO sample intelligently uniformly in each cell of a 9x9 grid
            
            Ncorners1 = self.grid_n_corner_detection(src_frame[:,:,:center_point[0],center_point[1]:], max(0,to_resample[1])) #TODO sample intelligently uniformly in each cell of a 9x9 grid
            Ncorners1[:,1] += center_point[1]

            Ncorners2 = self.grid_n_corner_detection(src_frame[:,:,center_point[0]:,:center_point[1]], max(0,to_resample[2])) #TODO sample intelligently uniformly in each cell of a 9x9 grid
            Ncorners2[:,0] += center_point[0]

            Ncorners3 = self.grid_n_corner_detection(src_frame[:,:,center_point[0]:,center_point[1]:], max(0,to_resample[3])) #TODO sample intelligently uniformly in each cell of a 9x9 grid
            Ncorners3[:,0] += center_point[0]
            Ncorners3[:,1] += center_point[1]


            # print("Nbr of points resampled / kept / repartition: ", (nbr_new_keypoint), init_queries_first_frame.shape[0], to_resample)

            #print("points kept during resampling : ",init_queries_first_frame)
            
            # add the prior = the keypoint still visible from the last tracks
            queries_2d_coords = torch.cat((init_queries_first_frame.to('cpu'), Ncorners), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners1), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners2), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners3), dim=0)

            # vis_harris(queries_2d_coords, src_frame)

            src_steps_tensor = torch.full((queries_2d_coords.shape[0], 1), src_step)
            src_corners = torch.cat((src_steps_tensor,queries_2d_coords), dim=1) #coordonate contain src_frame_index
            src_corners = torch.stack([src_corners], dim=0)
            src_points.append(src_corners)

            #src_points[0].shape torch.Size([1, 64, 3])
            src_points = torch.cat(src_points, dim=1)

            #src_points = torch.Size([1, 64, 3]) #3 = (frame=0, height_y width_x)


            _, _ = self.modelOnline(video_chunck.to('cuda'), src_points.to('cuda'), is_first_step=True)
            self.OnlineCoTracker_initialized = True

    def merge_accumulated_tracks(self, tracks, track_overlap=4, matching_threshold = 15):

        tracks = tracks.to('cpu')

        if self.accumulated_tracks is None:
            return tracks
        
        #print("merge_accumulated_tracks : tracks.shape", tracks.shape)
        #print("merge_accumulated_tracks : self.accumulated_tracks.shape", self.accumulated_tracks.shape)

        #if self.accumulated_tracks_end_dict is None:
        #    self.accumulated_tracks_end_dict = {}
        #    for i in range(self.accumulated_tracks.shape[2]):
        #        self.accumulated_tracks_end_dict[self.accumulated_tracks[0,-track_overlap,i,:2]] = i # save index of every end(just before overlap) of track accumulated
        #        print(self.accumulated_tracks[0,-track_overlap,i,:])



        increase_track_size = tracks.shape[1]-track_overlap
        p3d = (0, 0, 0, 0, 0, increase_track_size, 0, 0) # (0, 1, 2, 1, 3, 3) # pad by (0, 1)=last dim padding, (2, 1)=second to last dim padding, and (3, 3)
        out_tracks = torch.nn.functional.pad(self.accumulated_tracks, p3d, "constant", 0)
        start_of_new_track = tracks.shape[1] #start position from right of the new track




        acumulated_track_end = self.accumulated_tracks[0,-track_overlap,:,:2]
        new_tracks_start = tracks[0,0,:,:2]
        pairwise_norm = torch.cdist(acumulated_track_end, new_tracks_start, p=2) # p=2 => = eucnlidean norm
        new_to_acumulated= torch.argmin(pairwise_norm, dim=0)
        acumulate_to_new = torch.argmin(pairwise_norm, dim=1)
        #print("pairwise_norm", pairwise_norm)



        counter_extended, counter_created = 0,0
        for tr in range(tracks.shape[2]):
            #print("pairwise_norm", pairwise_norm[new_to_acumulated[tr],tr])
            if pairwise_norm[new_to_acumulated[tr],tr] < matching_threshold:
                counter_extended +=1
                out_tracks[:,-start_of_new_track:,new_to_acumulated[tr],:] = tracks[:, :, tr, :]
            else:
                counter_created +=1
                p3d = (0, 0, 0, 1, 0, 0, 0, 0)
                out_tracks = torch.nn.functional.pad(out_tracks, p3d, "constant", 0)
                out_tracks[:,-start_of_new_track:,-1,:] = tracks[:, :, tr, :]





        #print("-------------------------------")

        #count1, count2 = 0,0
        #for j in range(tracks.shape[2]): #for every track
        #    print(tracks[0, 0, j, :])
        #    if tracks[0, 0, j, :2] in self.accumulated_tracks_end_dict:
        #        count1 +=1
        #        original_track = self.accumulated_tracks_end_dict[tracks[0, 0, j, :2]]
        #        out_tracks[:,-start_of_new_track:,original_track,:] = tracks[:, :, j, :]
        #    else:
        #        count2 +=1
        #        p3d = (0, 0, 0, 1, 0, 0, 0, 0)
        #        out_tracks = torch.nn.functional.pad(out_tracks, p3d, "constant", 0)
        #        out_tracks[:,-start_of_new_track:,-1,:] = tracks[:, :, j, :]

        # print("merge_accumulated_tracks : out_tracks.shape, track extended, track created", out_tracks.shape, counter_extended, counter_created)
        return out_tracks
        #track_accumulator[:-4] #last four frames overlap continuity was made on the first of this last frame



    def get_tracks_at_motion_boundaries_online_droid(self, data, num_tracks=2000, sim_tracks=2000,
                                        **kwargs):

        N, S = num_tracks, sim_tracks
        # start = time.time()
        video_chunck = data["video_chunk"]
        #print("get_tracks_at_motion_boundaries_online_droid : video_chunck.shape", video_chunck.shape)

        B, T, _, H, W = video_chunck.shape

        backward_tracking = False
        flip = False

        if flip:
            video_chunck = video_chunck.flip(dims=[1])

        # Track batches of points
        tracks = []
        cache_features = True


        if not self.OnlineCoTracker_initialized:
            self.accumulated_tracks = None
            self.init_sampl_func(data, num_tracks=num_tracks, sim_tracks=sim_tracks)
            return {"tracks": tracks}


        lost_nbr_of_frame_not_visible = 5
        threshold_minimum_nbr_visible_tracks_wanted = (3*S)//4


        traj, vis = self.modelOnline(video_chunck, None, is_first_step=False)
        tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
        cache_features = False
        tracks = torch.cat(tracks, dim=2)


        vis_lost_window = vis[0, -lost_nbr_of_frame_not_visible:,:]
        tracks_not_lost_vis, _ = torch.max(vis_lost_window, 0) #dim 0 is the time(frames)

        tracks = self.merge_accumulated_tracks(tracks)
        if torch.sum(tracks_not_lost_vis)<threshold_minimum_nbr_visible_tracks_wanted:
            self.accumulated_tracks = tracks
            self.accumulated_tracks_end_dict = None
            tracks_not_lost_mask = tracks_not_lost_vis==1
            queries_kept = traj[0,-1, tracks_not_lost_mask,:]
            self.init_sampl_func(data, num_tracks=num_tracks, sim_tracks=sim_tracks, init_queries_first_frame=queries_kept)

        if flip:
            tracks = tracks.flip(dims=[1])
        # end = time.time()
        # print('runtime for tracking:', end - start)

        return {"tracks": tracks}


    def init_motion_boundaries(self, data, num_tracks=8192, sim_tracks=2048,
                                                     sample_mode="first",
                                                     **kwargs): 

        N, S = 64, 64  # num_tracks, sim_tracks
        start = time.time()
        video_chunck = data["video_chunk"]

        B, T, _, H, W = video_chunck.shape
        assert T>1 #require at least two frame to get motion boundaries (the motion boundaries are computed between frame 0 and 1

        assert T<3 #TO be removed but why is this function run with a long video ? (TODO : Is RAFT to good enough with two frames ?)
        if sample_mode == "all":
            samples_per_step = [S // T for _ in range(T)]
            samples_per_step[0] += S - sum(samples_per_step)
            backward_tracking = True
            flip = False
        elif sample_mode == "first":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False #TODO changed this does it impact ? also for the main tracking funcion
            flip = False
        elif sample_mode == "last":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = True
        else:
            raise ValueError(f"Unknown sample mode {sample_mode}")

        if flip:
            video_chunck = video_chunck.flip(dims=[1])

        motion_boundaries = {}  #TODO consider saving it to the state if function need to be recalled, for now save memory
        src_points = []

        for src_step, src_samples in enumerate(samples_per_step):
            if src_samples == 0:
                continue
            if not src_step in motion_boundaries:
                tgt_step = src_step - 1 if src_step > 0 else src_step + 1
                data = {"src_frame": video_chunck[:, src_step], "tgt_frame": video_chunck[:, tgt_step]}
                pred = self.optical_flow_estimator(data, mode="motion_boundaries", **kwargs)
                motion_boundaries[src_step] = pred["motion_boundaries"]
            src_boundaries = motion_boundaries[src_step]
            src_points.append(sample_points(src_step, src_boundaries, src_samples))

        src_points = torch.cat(self.src_points, dim=1)

        _, _ = self.modelOnline(video_chunck, self.src_points, is_first_step=True)
        self.OnlineCoTracker_initialized = True


    def get_tracks_at_motion_boundaries(self, data, num_tracks=8192, sim_tracks=2048, sample_mode="all",
                                        **kwargs):
        num_tracks, sim_tracks = 64, 64
        # start = time.time()
        video = data["video"]
        N, S = num_tracks, sim_tracks
        B, T, _, H, W = video.shape
        assert N % S == 0

        # Define sampling strategy
        if sample_mode == "all":
            samples_per_step = [S // T for _ in range(T)]
            samples_per_step[0] += S - sum(samples_per_step)
            backward_tracking = True
            flip = False
        elif sample_mode == "first":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = False
        elif sample_mode == "last":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = True
        else:
            raise ValueError(f"Unknown sample mode {sample_mode}")

        if flip:
            video = video.flip(dims=[1])

        backward_tracking = False

        # Track batches of points
        tracks = []
        motion_boundaries = {}
        cache_features = True
        for _ in tqdm(range(N // S), desc="Track batch of points", leave=False):
            src_points = []
            for src_step, src_samples in enumerate(samples_per_step):
                if src_samples == 0:
                    continue
                if not src_step in motion_boundaries:
                    tgt_step = src_step - 1 if src_step > 0 else src_step + 1
                    data = {"src_frame": video[:, src_step], "tgt_frame": video[:, tgt_step]}
                    pred = self.optical_flow_estimator(data, mode="motion_boundaries", **kwargs)
                    motion_boundaries[src_step] = pred["motion_boundaries"]
                src_boundaries = motion_boundaries[src_step]
                src_points.append(sample_points(src_step, src_boundaries, src_samples))

            src_points = torch.cat(src_points, dim=1)
            traj, vis = self.model(video, src_points, backward_tracking, cache_features)
            tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
            cache_features = False
        tracks = torch.cat(tracks, dim=2)

        if flip:
            tracks = tracks.flip(dims=[1])
        # end = time.time()
        # print('runtime for tracking:', end - start)

        return {"tracks": tracks}

    def get_flow_from_last_to_first_frame(self, data, sim_tracks=2048, **kwargs):
        video = data["video"]
        video = video.flip(dims=[1])
        src_step = 0  # We have flipped video over temporal axis so src_step is 0
        B, T, C, H, W = video.shape
        S = sim_tracks
        backward_tracking = False
        cache_features = True
        flow = get_grid(H, W, shape=[B]).cuda()
        flow[..., 0] = flow[..., 0] * (W - 1)
        flow[..., 1] = flow[..., 1] * (H - 1)
        alpha = torch.zeros(B, H, W).cuda()
        mask = torch.ones(H, W)
        pbar = tqdm(total=H * W // S, desc="Track batch of points", leave=False)
        while torch.any(mask):
            points, (i, j) = sample_mask_points(src_step, mask, S)
            idx = i * W + j
            points = points.cuda()[None].expand(B, -1, -1)

            traj, vis = self.model(video, points, backward_tracking, cache_features)
            traj = traj[:, -1]
            vis = vis[:, -1].float()

            # Update mask
            mask = mask.view(-1)
            mask[idx] = 0
            mask = mask.view(H, W)

            # Update flow
            flow = flow.view(B, -1, 2)
            flow[:, idx] = traj - flow[:, idx]
            flow = flow.view(B, H, W, 2)

            # Update alpha
            alpha = alpha.view(B, -1)
            alpha[:, idx] = vis
            alpha = alpha.view(B, H, W)

            cache_features = False
            pbar.update(1)
        pbar.close()
        return {"flow": flow, "alpha": alpha}
