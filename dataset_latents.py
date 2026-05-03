from torch.utils.data import Dataset, DataLoader
import torch

class LatentDataset(Dataset):
  def __init__(self, pt_file, num_clips, region): 
    data = torch.load(pt_file, weights_only=False)

    self.embeddings = torch.tensor([e['embedding'] for e in data['embeddings']])

  def __len__(self):
    return len(self.embeddings)
  
  def __getitem__(self, idx):
    return self.embeddings[idx]