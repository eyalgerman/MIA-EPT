import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class ChallengeDataset(Dataset):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

        challenge_points_path = os.path.join(self.base_dir, "challenge_with_id.csv")
        challenge_points_label = os.path.join(self.base_dir, "challenge_label.csv")

        self.challenge_points = pd.read_csv(challenge_points_path)
        if not os.path.exists(challenge_points_label):
            self.challenge_labels = None
        else:
            self.challenge_labels = pd.read_csv(challenge_points_label)
        # keep only the columns that not end with "_id"
        self.challenge_points = self.challenge_points.loc[:, ~self.challenge_points.columns.str.endswith('_id')]

    def __len__(self) -> int:
        return len(self.challenge_points)
    
    def __getitem__(self, idx) -> torch.Tensor:
        x = self.challenge_points.iloc[idx].to_numpy()
        return torch.from_numpy(x)

    def get_item_with_label(self, idx):
        x = self.challenge_points.iloc[idx].to_numpy()
        if self.challenge_labels is None:
            return torch.from_numpy(x), None
        y = self.challenge_labels.iloc[idx].to_numpy()
        return torch.from_numpy(x), torch.from_numpy(y)


class SyntheticDataset(Dataset):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

        trans_synthetic_path = os.path.join(self.base_dir, "trans_synthetic.csv")
        if not os.path.exists: raise FileNotFoundError(f"Trans Synthetic Points Path: {trans_synthetic_path} not found.")
        self.trans_synthetic_points = pd.read_csv(trans_synthetic_path)
        # keep only the columns that not end with "_id"
        self.trans_synthetic_points = self.trans_synthetic_points.loc[:, ~self.trans_synthetic_points.columns.str.endswith('_id')]


    def __len__(self) -> int:
        return len(self.trans_synthetic_points)

    def __getitem__(self, idx) -> torch.Tensor:
        x = self.trans_synthetic_points.iloc[idx].to_numpy()
        return torch.from_numpy(x)

def get_challenge_points(base_dir: Path) -> torch.Tensor: 
    dataset = ChallengeDataset(base_dir)

    loader = DataLoader(dataset, batch_size=200)
    challenge_points = next(iter(loader))

    return challenge_points


