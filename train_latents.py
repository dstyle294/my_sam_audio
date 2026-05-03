from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
import torch

train_data = LatentDataset("/home/s.dalal.334/SAM/embeddings/train/50_HSN_latents.pt", 50, "HSN")

test_data = LatentDataset("/home/s.dalal.334/SAM/embeddings/test/100_HSN_latents.pt", 100, "HSN")

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)



