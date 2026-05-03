import torch.nn as nn

class BirdSetClassifier(nn.Module):
  def __init__(self, region, input_dim=256):