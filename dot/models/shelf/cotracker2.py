from torch import nn

from .cotracker2_utils.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor


class CoTracker2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerPredictor(args.patch_size, args.wind_size)

    def forward(self, video, queries, backward_tracking, cache_features=False):
        return self.model(video, queries=queries, backward_tracking=backward_tracking, cache_features=cache_features)


class CoTracker2Online(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = CoTrackerOnlinePredictor(args.patch_size, args.wind_size)

    def forward(self, video_chunk, queries, is_first_step=False):
        return self.model(video_chunk, is_first_step=False, queries=queries,)

