from tqdm import tqdm
import torch
from torch import nn
import time
import cv2
import numpy as np

from .optical_flow import OpticalFlow
from .shelf import CoTracker2Online
from thirdparty.DOT.dot.utils.io import read_config
from thirdparty.DOT.dot.utils.torch import sample_points, sample_mask_points, get_grid

import matplotlib.pyplot as plt
from scipy import ndimage
import math



'''
Method decdicated for the visualisation of a list of point on an image(frame extracted from the video of the scene)
The points(Corner) are displayed as red dots
'''
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

    '''
    We added an argument to setup the class relying on Cotracker Online instead of normal Cotracker which process the complete video as one input
    '''
    def __init__(self,  height, width, tracker_config, tracker_path, estimator_config, estimator_path, isOnline=False):
        super().__init__()
        model_args = read_config(tracker_config)

        self.cnt = 0

        sampling_inititization_functions  = {
            'harris' : self.init_harris,
            'grid' : self.init_grid
            }
        self.init_sampl_func = sampling_inititization_functions['harris'] # the function called when we want to get more point in the tracking

        self.OnlineCoTracker_initialized = False
        self.modelOnline = CoTracker2Online(model_args)
        if tracker_path is not None:
            device = next(self.modelOnline.parameters()).device
            self.modelOnline.load_state_dict(torch.load(tracker_path, map_location=device), strict=False)
        self.optical_flow_estimator = OpticalFlow(height, width, estimator_config, estimator_path)

    def forward(self, data, mode, **kwargs):
        self.cnt += 1
        if mode == "tracks_online_droid":
            return self.get_tracks_online_droid(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")
        
    '''
    Simple grid key point extraction, aligning a grid of n(n_keypoints) to return each intersection as a tensor(list) of points
    '''
    def grid_n_corner_detection(self, src_frame_tensor, n_keypoints):
        if n_keypoints <= 0:
            return torch.empty((0,2))
        b, c, height, width = src_frame_tensor.shape
        num = int(math.sqrt(n_keypoints))

        grid_a = np.linspace(1, width - 1, num)
        grid_b = np.linspace(1, height - 1, num)
        grid_x, grid_y = np.meshgrid(grid_a, grid_b)
        grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T.astype(np.int16)
        x, y = 8, 8
        gap = 20
        while n_keypoints > grid_points.shape[0]:
            grid_points = np.vstack((grid_points, np.array([x, y], dtype=np.int16)))
            x += gap
            y += gap
        return torch.tensor(grid_points)

    '''
    Convert the frame to grey using human eye perception value and compute the harris corner strengh at each pixels
    '''
    def dst_harris_computation(self, src_frame_tensor):
        src_frame_tensor = src_frame_tensor.to('cpu')
        r, g, b = src_frame_tensor[:, 0, :, :], src_frame_tensor[:, 1, :, :], src_frame_tensor[:, 2, :, :]
        grayscale_tensor = 0.2989 * r + 0.5870 * g + 0.1140 * b  # according to the human eye may not be best here
        grayscale_tensor = torch.permute(grayscale_tensor, (1, 2, 0))
        grayscale_numpy = grayscale_tensor.numpy()
        dst = cv2.cornerHarris(grayscale_numpy, 2, 3, 0.04)
        return dst

    '''
    helper method of tracks_online_droid
    Given a video, apply on the first frame of the video (src_step=0):
    use harris on the first frame to extract keypoint 
    param nbr_grid_cell_width, nbr_grid_cell_height = harris tend to group keypoints in cluster around the same region of the image, 
    some strategy such as maximum suppression exist here because we are interpoleting the flow, we wanted a 
    distribution of keypoint relatively unifrom though the image, for that we divide the image in grid cell 
    and divide the total nbr of point though resample between each cell, for each we extract the point in then with the heighest harris score. 
    '''
    def init_harris(self, data, num_tracks_max=8192, sim_tracks=512,
                    sample_mode="first", init_queries_first_frame=torch.empty((0, 2)).to('cuda'),
                    nbr_grid_cell_width=20, nbr_grid_cell_height=20,
                    **kwargs):

        video_chunck = data["video_chunk"]

        B, T, _, H, W = video_chunck.shape # extract from the shape the batch size, the 
        assert T >= 1  # require at least two frame to get motion boundaries (the motion boundaries are computed between frame 0 and 1

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


        # add the prior = the keypoint still visible from the last tracks


        src_steps_tensor = torch.full((queries_2d_coords.shape[0], 1), src_step)
        src_corners = torch.cat((src_steps_tensor, queries_2d_coords), dim=1)  # coordonate contain src_frame_index
        src_corners = torch.stack([src_corners], dim=0)
        src_points.append(src_corners)

        src_points = torch.cat(src_points, dim=1)


        _, _ = self.modelOnline(video_chunck.to('cuda'), src_points.to('cuda'), is_first_step=True)
        self.OnlineCoTracker_initialized = True



    '''
    helper method of tracks_online_droid
    Alternative to harris, here we simply extract the key point on a grid in four cell of the image  

    '''

    def init_grid(self, data, num_tracks_max=512, sim_tracks=512,
                                                        sample_mode="first", init_queries_first_frame=torch.empty((0, 2)).to('cuda'),
                                                        **kwargs):

            N, S = num_tracks_max, sim_tracks  # num_tracks, sim_tracks
            video_chunck = data["video_chunk"]

            B, T, _, H, W = video_chunck.shape
            assert T>=1 #require at least two frame to get motion boundaries (the motion boundaries are computed between frame 0 and 1
            

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


            for i in range(difference_left):
                to_resample[i%4] +=1

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


            
            # add the prior = the keypoint still visible from the last tracks
            queries_2d_coords = torch.cat((init_queries_first_frame.to('cpu'), Ncorners), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners1), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners2), dim=0)
            queries_2d_coords = torch.cat((queries_2d_coords, Ncorners3), dim=0)


            src_steps_tensor = torch.full((queries_2d_coords.shape[0], 1), src_step)
            src_corners = torch.cat((src_steps_tensor,queries_2d_coords), dim=1) #coordonate contain src_frame_index
            src_corners = torch.stack([src_corners], dim=0)
            src_points.append(src_corners)

            src_points = torch.cat(src_points, dim=1)



            _, _ = self.modelOnline(video_chunck.to('cuda'), src_points.to('cuda'), is_first_step=True)
            self.OnlineCoTracker_initialized = True

    '''
    helper method of tracks_online_droid
    Due to a size constraint and an initialization constraint of Cotracker asking for tracks starting point at the beginning, we decided to maintain two sets of track one being the currenctly beeing extended  by cotracker and one save all tracks seem until now,
    to get the final set of tracker we use this function to stack all tracks in a matrix extended vertically in this way :
    1. We pad the old track mmatrix to the size of the extended tracks (total nbr of frame)
    2. cut the track to remove the overlap
    3. compute the distance between the end of each old track and start of each new one 
    4. If the end of an old track is within matching_threshold of a new one then we bound them togerther by adding the new track on the corresponding old track line 
    5. else we just append it to the end of the tracks matrix as a track not visible for the old frames 

    '''
    def merge_accumulated_tracks(self, tracks, track_overlap=4, matching_threshold = 15):


        tracks = tracks.to('cpu')

        if self.accumulated_tracks is None:
            return tracks
        
        #1.
        increase_track_size = tracks.shape[1]-track_overlap
        p3d = (0, 0, 0, 0, 0, increase_track_size, 0, 0) # (0, 1, 2, 1, 3, 3) # pad by (0, 1)=last dim padding, (2, 1)=second to last dim padding, and (3, 3)
        out_tracks = torch.nn.functional.pad(self.accumulated_tracks, p3d, "constant", 0)
        start_of_new_track = tracks.shape[1] #start position from right of the new track

        #2.
        acumulated_track_end = self.accumulated_tracks[0,-track_overlap,:,:2] #

        #3. 
        new_tracks_start = tracks[0,0,:,:2]
        pairwise_norm = torch.cdist(acumulated_track_end, new_tracks_start, p=2) # p=2 => = eucnlidean norm
        new_to_acumulated= torch.argmin(pairwise_norm, dim=0) # dim= 1 would correspond to acumulate_to_new


        #4.
        for tr in range(tracks.shape[2]):
            if pairwise_norm[new_to_acumulated[tr],tr] < matching_threshold:
                out_tracks[:,-start_of_new_track:,new_to_acumulated[tr],:] = tracks[:, :, tr, :]
            else:
                p3d = (0, 0, 0, 1, 0, 0, 0, 0)
                out_tracks = torch.nn.functional.pad(out_tracks, p3d, "constant", 0)
                out_tracks[:,-start_of_new_track:,-1,:] = tracks[:, :, tr, :]

        return out_tracks


    '''
    In data we process the new video_chunk of size 4 correspondig to the window size of the current training of cotracker 
    If we execute the method for the first time, we call init_sampl_func to generate new keypoint to track and initilize online Cotracker with this new points 
    Tunable parameter (lost_nbr_of_frame_not_visible, threshold_minimum_nbr_visible_tracks_wanted) = threshold corresponding to the nbr of keypoint we require to be visible at least one time in the last lost_nbr_of_frame_not_visible frame 
    Algo: 
        1. Run cotracker to extend the tracks 
        2. call merge_accumulated_tracks to get the track compose of the old tacks and the new extended one this is performed everytime to account for possible complete variations of the new tracks depending on the cotracker window size 
        3. Check if the require nbr rof tracks are still visible, if not
            replace the old track by the output of merge_accumulated_tracks(old track + completed new ones)
            Reinitialize a new instance of cotracker with all track not considered as lost in the last finsihed run 
    '''
    def get_tracks_online_droid(self, data, num_tracks=512, sim_tracks=512,
                                        **kwargs):

        N, S = num_tracks, sim_tracks
        video_chunck = data["video_chunk"]

        B, T, _, H, W = video_chunck.shape

        flip = False

        if flip:
            video_chunck = video_chunck.flip(dims=[1])

        # Track batches of points
        tracks = []


        if not self.OnlineCoTracker_initialized:
            self.accumulated_tracks = None # will contain all the track from group of frame between the first one seen and the last gotten 
            self.init_sampl_func(data, num_tracks=num_tracks, sim_tracks=sim_tracks) 
            return {"tracks": tracks}


        lost_nbr_of_frame_not_visible = 5
        threshold_minimum_nbr_visible_tracks_wanted = (7*S)//8 # threshold corresponding to the nbr of keypoint we require to be visible at least one time in the last lost_nbr_of_frame_not_visible frame 


        traj, vis = self.modelOnline(video_chunck, None, is_first_step=False)
        tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
        tracks = torch.cat(tracks, dim=2)


        vis_lost_window = vis[0, -lost_nbr_of_frame_not_visible:,:]
        tracks_not_lost_vis, _ = torch.max(vis_lost_window, 0) #dim 0 is the time(frames)

        tracks = self.merge_accumulated_tracks(tracks)
        if torch.sum(tracks_not_lost_vis)<threshold_minimum_nbr_visible_tracks_wanted:
            self.accumulated_tracks = tracks
            tracks_not_lost_mask = tracks_not_lost_vis==1
            queries_kept = traj[0,-1, tracks_not_lost_mask,:]
            self.init_sampl_func(data, num_tracks=num_tracks, sim_tracks=sim_tracks, init_queries_first_frame=queries_kept)

        if flip:
            tracks = tracks.flip(dims=[1])

        return {"tracks": tracks}

