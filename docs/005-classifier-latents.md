## Motivation
Class to serve as the linear classifier for latents.

## Conventions
There are options for the types of pooling:
* **flatten**: multiply time and channel dimension to generate a 16,000 sized input. This leads to more parameters, which makes the model more conducive to overfitting
* **mean**: pool by taking the mean of the channel dimension
* **max**: pool by taking the max of the channel dimension
* **max_mean**: pool by concatenating the mean then max pool of the channel dimension.

Batch Normalization is used, and leads to a running mean which is used during inference. This could lead to issues in domain shift problems. This is because the running mean computed from the training distribution is used for the test distribution - but these distributions are fundamentally different.

