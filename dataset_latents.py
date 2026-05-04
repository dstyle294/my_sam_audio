from torch.utils.data import Dataset, DataLoader
import torch
from datasets import load_dataset, ClassLabel, Sequence
from copy import copy
import numpy as np

def one_hot_encode(labels, classes):
    one_hot = np.zeros(len(classes))
    for label in labels:
        one_hot[label] = 1
    return np.array(one_hot, dtype=float)

def one_hot_encode_ds_wrapper(row, class_list):
    row["labels"] = one_hot_encode(row["labels"], class_list)
    return row

class LatentDataset(Dataset):
  def __init__(self, pt_file, num_clips, region, split): 
    data = torch.load(pt_file, weights_only=False)

    self.embeddings = torch.tensor([e['embedding'] for e in data['embeddings']])

    ds = load_dataset("DBD-research-group/BirdSet", region, trust_remote_code=True)
  

    class_list = ds["train"].features["ebird_code"].names

    self.num_classes = len(class_list)

    ds = ds[split].select(range(num_clips))

    multilabel_class_label = Sequence(ClassLabel(names=class_list))

    self.labels = copy(ds["ebird_code_multilabel"])

    self.labels = [one_hot_encode(label, class_list).tolist() for label in self.labels]

    self.embeddings = torch.as_tensor(self.embeddings).float()
    self.labels = torch.as_tensor(self.labels).int()

  def __len__(self):
    return len(self.embeddings)
  
  def __getitem__(self, idx):
    return self.embeddings[idx], self.labels[idx]