from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
import torch
from latent_classifier import BirdSetClassifier

train_data = LatentDataset("/home/s.dalal.334/SAM/embeddings/train/50_HSN_latents.pt", 50, "HSN", "train")

test_data = LatentDataset("/home/s.dalal.334/SAM/embeddings/test/100_HSN_latents.pt", 100, "HSN", "test_5s")

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)


model = BirdSetClassifier(num_classes=train_data.num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(20):
  for features, labels in train_dataloader:
    logits = model(features)
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for features, labels in test_dataloader:
  logits = model(features)
  loss = criterion(logits, labels)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  print(loss)


  



