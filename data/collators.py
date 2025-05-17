import torch
from torch.utils.data import Dataset

class PerceiverTrackDataset(Dataset):
    def __init__(self, num_samples, track_len=5000):
        self.N = num_samples
        self.track_len = track_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        track_nt = torch.randint(0, 5, (self.track_len,), dtype=torch.long)
        logcov = torch.rand(self.track_len) * 10.0
        target = torch.rand(1) * 10.0
        return track_nt, logcov, target

class PerceiverMPRADataset(Dataset):
    def __init__(self, num_samples, mpra_len=150):
        self.N = num_samples
        self.mpra_len = mpra_len

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        mpra_nt = torch.randint(0, 5, (self.mpra_len,), dtype=torch.long)
        activity = torch.rand(1) * 10.0 # Generate activity, scaled 0-10
        # The target for MPRA prediction is the activity itself
        return mpra_nt, activity, activity 