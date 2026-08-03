from dataset_latents import LatentDataset
from torch.utils.data import DataLoader
from train_latents import train_and_evaluate
import argparse
import csv
import itertools
import time

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", required=True, help="Path to saved train latent dataset")
parser.add_argument("--test_path", required=True, help="Path to saved test latent dataset")
parser.add_argument("--train_batch_size", required=False, help="Batch size for training", default=8, type=int)
parser.add_argument("--test_batch_size", required=False, help="Batch size for testing", default=8, type=int)

# sweep dimensions -- comma-separated lists, e.g. --poolings mean_max,flatten
parser.add_argument("--poolings", required=False, default="mean_max",
                     help="Comma-separated list of pooling modes to sweep, e.g. mean_max,flatten,max,mean,gru")
parser.add_argument("--num_epochs_list", required=False, default="20",
                     help="Comma-separated list of epoch counts to sweep, e.g. 10,20")
parser.add_argument("--lrs", required=False, default="5e-4",
                     help="Comma-separated list of learning rates to sweep, e.g. 1e-3,5e-4,1e-4")
parser.add_argument("--hidden_dims", required=False, default="1024",
                     help="Comma-separated list of hidden layer sizes to sweep, e.g. 256,1024")

# fixed across the whole sweep, not part of the grid
parser.add_argument("--norm_type", required=False, default="layernorm", choices=["layernorm", "batchnorm"])
parser.add_argument("--pos_weight_clamp", required=False, default=20, type=int,
                     help="Caps the max per-class pos_weight. 0 disables pos_weight entirely.")

parser.add_argument("--output_csv", required=False, default="sweep_results.csv")


def parse_list(s, cast):
    return [cast(v.strip()) for v in s.split(",") if v.strip() != ""]


def main(args):
    train_ds = LatentDataset(args.train_path)
    test_ds = LatentDataset(args.test_path)

    # dataset/loaders built once and reused across every combo -- only the
    # model (architecture + hidden size) and training hyperparameters change
    train_loader = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    print(f"num_classes: {train_ds.num_classes}")
    print(f"latent shape: {train_ds.latents.shape}")
    print(f"label shape:  {train_ds.labels.shape}")

    poolings = parse_list(args.poolings, str)
    num_epochs_list = parse_list(args.num_epochs_list, int)
    lrs = parse_list(args.lrs, float)
    hidden_dims = parse_list(args.hidden_dims, int)

    combos = list(itertools.product(poolings, num_epochs_list, lrs, hidden_dims))
    print(f"\nRunning {len(combos)} combinations...\n")

    results = []
    for i, (pooling, num_epochs, lr, hidden_dim) in enumerate(combos):
        start = time.time()
        print(f"[{i+1}/{len(combos)}] pooling={pooling} epochs={num_epochs} lr={lr} hidden_dim={hidden_dim}")

        # same train_and_evaluate() that train_latents.py's single-run mode
        # uses -- plot=False skips the confusion matrix, verbose=False skips
        # per-epoch prints (the sweep prints its own per-combo progress instead)
        result = train_and_evaluate(
            train_ds, train_loader, test_loader, train_ds.num_classes,
            pooling=pooling, hidden_dim=hidden_dim, norm_type=args.norm_type,
            num_epochs=num_epochs, lr=lr, pos_weight_clamp=args.pos_weight_clamp,
            plot=False, verbose=False,
        )
        elapsed = time.time() - start
        result["elapsed_sec"] = round(elapsed, 1)
        results.append(result)

        print(f"    cmAP={result['cmAP']:.4f}  ROCAUC={result['ROCAUC']:.4f}  "
              f"loss={result['avg_test_loss']:.4f}  ({elapsed:.1f}s)\n")

    fieldnames = list(results[0].keys())
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {args.output_csv}")

    print("\n--- Top results by cmAP ---")
    results_sorted = sorted(results, key=lambda r: r["cmAP"], reverse=True)
    for r in results_sorted[:10]:
        print(f"cmAP={r['cmAP']:.4f}  ROCAUC={r['ROCAUC']:.4f}  loss={r['avg_test_loss']:.4f}  "
              f"pooling={r['pooling']} epochs={r['num_epochs']} lr={r['lr']} hidden_dim={r['hidden_dim']}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)