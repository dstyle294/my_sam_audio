from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
import torch
from classifier_latents import BirdSetClassifier
import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", required=True, help="Path to saved train latent dataset")
parser.add_argument("--test_path", required=True, help="Path to saved test latent dataset")
parser.add_argument("--train_batch_size", required=False, help="Batch size for training", default=8)
parser.add_argument("--test_batch_size", required=False, help="Batch size for testing", default=8)

def main(args):
  train_ds = LatentDataset(args.train_path)
  test_ds  = LatentDataset(args.test_path)

  train_loader = DataLoader(train_ds, batch_size=int(args.train_batch_size), shuffle=True,  num_workers=4)
  test_loader  = DataLoader(test_ds,  batch_size=int(args.test_batch_size), shuffle=False, num_workers=4)

  print(f"num_classes: {train_ds.num_classes}")
  print(f"latent shape: {train_ds.latents.shape}")  # [E, C, T]
  print(f"label shape:  {train_ds.labels.shape}")   # [E, N_C]

  model = BirdSetClassifier(num_classes=train_ds.num_classes)

  optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

  criterion = torch.nn.BCEWithLogitsLoss()

  for epoch in range(20):
    for features, labels in train_loader:
      logits = model(features)
      loss = criterion(logits, labels.float())
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    print(f"Epoch {epoch + 1} done out of 20")

  model.eval() # turns off dropout, batch normalization
  test_loss = 0
  all_preds = torch.tensor([])
  all_labels = torch.tensor([])


  with torch.no_grad(): # 2. Disable gradient tracking
    for features, labels in test_loader:
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
  avg_loss = test_loss / len(test_ds)

  # 4. Calculating ROCAUC + cMAP

  get_cmAP = metrics.cmAP(train_ds.num_classes)
  get_ROCAUC = metrics.ROCAUC(train_ds.num_classes)

  cmAP = get_cmAP(all_preds, all_labels)
  ROCAUC = get_ROCAUC(all_preds, all_labels)

  print(f"cmAP = {cmAP}")
  print(f"ROCAUC = {ROCAUC}")
  print(f"Average Test Loss = {avg_loss}")


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)