from torch.utils.data import Dataset, DataLoader
import torch


class LatentDataset(Dataset):
    def __init__(self, pt_file: str):
        data = torch.load(pt_file, weights_only=False)
        # data = {"latents": Tensor[E, C, T], "labels": Tensor[E, N_C]}
        self.latents = data["latents"].float()   # [E, C, T]
        self.labels  = data["labels"].float()    # [E, N_C]
        self.num_classes = self.labels.shape[1]

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]