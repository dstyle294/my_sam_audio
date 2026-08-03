import torch
import torch.nn as nn


class BirdSetClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=128, input_time=125,
                 pooling="mean_max", hidden_dim=1024):
        """
        pooling:
            "flatten"  -- original behavior, flattens [C, T] -> C*T (16,000-dim
                          for default shapes). Kept for comparison, not recommended --
                          massively overparameterized relative to a ~14k-example
                          training set (see the earlier discussion).
            "mean"     -- global average pool over time -> [C]
            "max"      -- global max pool over time -> [C]. Better suited to
                          brief, sparse events (bird calls) than mean pooling,
                          which dilutes a short call against mostly-silence.
            "mean_max" -- concatenate both -> [2*C]. Default: keeps the "peak
                          evidence of a call" signal from max while retaining
                          some sense of background level from mean.
        """
        super().__init__()
        self.pooling = pooling

        if pooling == "flatten":
            input_dim = input_channels * input_time
        elif pooling in ("mean", "max"):
            input_dim = input_channels
        elif pooling == "mean_max":
            input_dim = input_channels * 2
        else:
            raise ValueError(f"unknown pooling mode: {pooling}")

        self.bn = nn.BatchNorm1d(input_dim)  # normalizes the (pooled) SAM latents
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: [E, C, T]
        if self.pooling == "flatten":
            x = x.flatten(start_dim=1)
        elif self.pooling == "mean":
            x = x.mean(dim=2)
        elif self.pooling == "max":
            x = x.max(dim=2).values
        elif self.pooling == "mean_max":
            x = torch.cat([x.mean(dim=2), x.max(dim=2).values], dim=1)

        x = self.bn(x)
        return self.classifier(x)