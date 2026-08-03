from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
import torch
from classifier_latents import BirdSetClassifier
import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", required=True, help="Path to saved train latent dataset")
parser.add_argument("--test_path", required=True, help="Path to saved test latent dataset")
parser.add_argument("--train_batch_size", required=False, help="Batch size for training", default=8, type=int)
parser.add_argument("--test_batch_size", required=False, help="Batch size for testing", default=8, type=int)
parser.add_argument("--num_epochs", required=False, help="Number of epochs for training", default=20, type=int)
parser.add_argument("--lr", required=False, help="Learning rate", default=5e-4, type=float)
parser.add_argument("--pooling", required=False, help="Type of pooling", default="mean_max",
                     choices=["mean_max", "max", "mean", "flatten", "gru"])
parser.add_argument("--hidden_dim", required=False, help="Classifier hidden layer size", default=1024, type=int)
parser.add_argument("--norm_type", required=False, help="Normalization layer", default="layernorm",
                     choices=["layernorm", "batchnorm"])
parser.add_argument("--pos_weight_clamp", required=False, type=int, default=20,
                     help="Caps the max per-class pos_weight (uncapped num_neg/num_pos ratios can be "
                          "extreme for rare classes and destabilize training). 0 disables pos_weight entirely.")


def train_and_evaluate(train_ds, train_loader, test_loader, num_classes,
                        pooling="mean_max", hidden_dim=1024, norm_type="layernorm",
                        num_epochs=20, lr=5e-4, pos_weight_clamp=20,
                        plot=True, save_dir="plots/confusion_matrix", verbose=True):
    """
    Builds a fresh model, trains it for num_epochs, evaluates on test_loader, and
    returns a dict of results. This is the single source of truth for train+eval
    logic -- both the single-run CLI (main(), below) and sweep_latents.py call
    this directly rather than duplicating the training loop.

    plot=False skips the confusion matrix entirely (used by the sweep, where
    computing/plotting one per combination would be wasteful).
    verbose=False suppresses per-epoch progress prints (used by the sweep, which
    prints its own per-combination progress instead).
    """
    model = BirdSetClassifier(
        num_classes=num_classes,
        pooling=pooling,
        hidden_dim=hidden_dim,
        norm_type=norm_type,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_weight = None
    if pos_weight_clamp > 0:
        num_pos = train_ds.labels.sum(dim=0)
        num_neg = len(train_ds) - num_pos
        # NOTE: clamp(max=...), not clamp(min=...) -- this caps the ceiling on
        # extreme per-class weights (rare classes can otherwise get weights in
        # the hundreds, which destabilizes training), it doesn't raise a floor
        pos_weight = (num_neg / num_pos.clamp(min=1)).clamp(max=pos_weight_clamp)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            logits = model(features)
            loss = criterion(logits, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1} done out of {num_epochs}")

    model.eval()
    test_loss = 0
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    with torch.no_grad():
        for features, labels in test_loader:
            logits = model(features)

            loss = criterion(logits, labels.float())
            test_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds = torch.cat((all_preds, probs), dim=0)
            all_labels = torch.cat((all_labels, labels), dim=0)

    all_labels = all_labels.int()
    avg_loss = test_loss / len(test_loader)

    get_cmAP = metrics.cmAP(num_classes)
    get_ROCAUC = metrics.ROCAUC(num_classes)

    cmAP = get_cmAP(all_preds, all_labels)
    ROCAUC = get_ROCAUC(all_preds, all_labels)

    if verbose:
        print(f"pred min:  {all_preds.min().item():.4f}")
        print(f"pred mean: {all_preds.mean().item():.4f}")
        print(f"pred max:  {all_preds.max().item():.4f}")
        print(f"cmAP = {cmAP}")
        print(f"ROCAUC = {ROCAUC}")
        print(f"Average Test Loss = {avg_loss}")

    if plot:
        get_ConfusionMatrix = metrics.ConfusionMatrix(num_classes)
        get_ConfusionMatrix(all_preds, all_labels)
        get_ConfusionMatrix.plot(save_dir=save_dir)

    return {
        "pooling": pooling,
        "num_epochs": num_epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "norm_type": norm_type,
        "pos_weight_clamp": pos_weight_clamp,
        "cmAP": cmAP,
        "ROCAUC": ROCAUC,
        "avg_test_loss": avg_loss,
        "pred_min": all_preds.min().item(),
        "pred_mean": all_preds.mean().item(),
        "pred_max": all_preds.max().item(),
    }


def main(args):
  train_ds = LatentDataset(args.train_path)
  test_ds  = LatentDataset(args.test_path)

  train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True,  num_workers=4)
  test_loader  = DataLoader(test_ds,  batch_size=args.test_batch_size, shuffle=False, num_workers=4)

  print(f"num_classes: {train_ds.num_classes}")
  print(f"latent shape: {train_ds.latents.shape}")  # [E, C, T]
  print(f"label shape:  {train_ds.labels.shape}")   # [E, N_C]

  train_and_evaluate(
      train_ds, train_loader, test_loader, train_ds.num_classes,
      pooling=args.pooling, hidden_dim=args.hidden_dim, norm_type=args.norm_type,
      num_epochs=args.num_epochs, lr=args.lr, pos_weight_clamp=args.pos_weight_clamp,
      plot=True, verbose=True,
  )


if __name__ == "__main__":
  args = parser.parse_args()
  main(args)