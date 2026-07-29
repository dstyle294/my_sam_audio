## Motivation
Class to serve as the linear classifier for latents.

## Conventions
Currently, to match sizes, the channel and time dimensions are flattened, giving a 125 * 128 = 16,000 size dimension. In the future, other options include pooling across the time dimension. 