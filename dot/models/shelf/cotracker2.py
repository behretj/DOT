from torch import nn

from .cotracker2_utils.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor

class CoTracker2Online(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerOnlinePredictor(args.patch_size, 8) #TODO change window size to 2

    def forward(self, video_chunk, queries, is_first_step=False):
        return self.model(video_chunk, is_first_step, queries=queries)

