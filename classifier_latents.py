import torch
import torch.nn as nn


class BirdSetClassifier(nn.Module):
    def __init__(self, num_classes, input_channels=128, input_time=125,
                 pooling="mean_max", hidden_dim=1024, norm_type="layernorm",
                 rnn_hidden=128, rnn_bidirectional=True):
        """
        pooling:
            "flatten"  -- original behavior, flattens [C, T] -> C*T (16,000-dim
                          for default shapes). Kept for comparison, not recommended --
                          massively overparameterized relative to a ~14k-example
                          training set.
            "mean"     -- global average pool over time -> [C]
            "max"      -- global max pool over time -> [C]. Better suited to
                          brief, sparse events (bird calls) than mean pooling,
                          which dilutes a short call against mostly-silence.
            "mean_max" -- concatenate both -> [2*C]. Keeps the "peak evidence
                          of a call" signal from max while retaining some sense
                          of background level from mean.
            "gru"      -- bidirectional GRU over the time axis, mean+max pooled
                          over its output sequence. Can capture temporal order
                          (call rhythm/note sequencing) that plain pooling can't,
                          at the cost of more training complexity.

        norm_type:
            "batchnorm" -- normalizes using batch statistics at train time and
                           accumulated running statistics at eval time -- these
                           can diverge if train/test come from different
                           distributions (e.g. different regions).
            "layernorm" -- default. Normalizes each example independently using
                           only its own features -- identical behavior in train
                           vs eval mode, no running statistic to mismatch against
                           a shifted test distribution.

        rnn_hidden / rnn_bidirectional: only used when pooling="gru".
        """
        super().__init__()
        self.pooling = pooling

        if pooling == "flatten":
            input_dim = input_channels * input_time
        elif pooling in ("mean", "max"):
            input_dim = input_channels
        elif pooling == "mean_max":
            input_dim = input_channels * 2
        elif pooling == "gru":
            self.rnn = nn.GRU(
                input_size=input_channels,
                hidden_size=rnn_hidden,
                batch_first=True,
                bidirectional=rnn_bidirectional,
            )
            gru_out_dim = rnn_hidden * (2 if rnn_bidirectional else 1)
            input_dim = gru_out_dim * 2  # mean+max pooled over the GRU's output sequence
        else:
            raise ValueError(f"unknown pooling mode: {pooling}")

        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(input_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(input_dim)
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

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
        elif self.pooling == "gru":
            seq = x.transpose(1, 2)          # [E, C, T] -> [E, T, C], GRU expects features last
            out, _ = self.rnn(seq)           # out: [E, T, gru_out_dim]
            x = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)

        x = self.norm(x)
        return self.classifier(x)