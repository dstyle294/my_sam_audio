from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
import torch
from latent_classifier import BirdSetClassifier
import metrics

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
    loss = criterion(logits, labels.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval() # turns off dropout, batch normalization
test_loss = 0
all_preds = torch.tensor([])
all_labels = torch.tensor([])


with torch.no_grad(): # 2. Disable gradient tracking
  for features, labels in test_dataloader:
    logits = model(features)
    
    # Calculate loss just for monitoring
    loss = criterion(logits, labels.float())
    test_loss += loss.item()

    # Collect results for metrics (CMAP/ROC-AUC)
    probs = torch.sigmoid(logits)
    all_preds = torch.cat((all_preds, probs), dim=0)
    all_labels = torch.cat((all_labels, labels), dim=0)

all_labels = all_labels.int()

# 3. Aggregate results
avg_loss = test_loss / len(test_dataloader)

# 4. Calculating ROCAUC + cMAP

get_cMAP = metrics.cMAP(train_data.num_classes)
get_ROCAUC = metrics.ROCAUC(train_data.num_classes)


cMAP = get_cMAP(all_preds, all_labels)
ROCAUC = get_ROCAUC(all_preds, all_labels)





  



