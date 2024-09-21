# Multilayer Perceptron (MLP)

This directory hosts a series of notebooks that cover various basic concepts related to MLPs.
The helper folder has basic functions that will be used across all notebooks to accomplish tasks such as:
1. Data loading
2. Training
3. Visualization

## Notebooks
In all the MLP notebooks I will work with the MNIST dataset.

Each notebook I try different configurations for the MLP and below I will document the model settings.

### Basic MLP - basic_mlp.ipynb
This notebook has a basic MLP with the following settings:

| Attribute               | Value  |
|-------------------------|--------|
| Number of hidden layers | 2      |
| Activation Function     | ReLU   |
| Optimizer               | SGD    |

It achieves the following results:

| Attribute               | Value  |
|-------------------------|--------|
| Train Accuracy          | 99.72% |
| Validation Accuracy     | 97.93% |
| Test Accuracy           | 97.76% |

### Basic MLP with Leaky ReLU - basic_mlp_leaky_relu.ipynb
This notebook has the same MLP as the previous notebook, the only difference is that I use Leaky ReLU as my activation function.

Model setup:

| Attribute               | Value      |
|-------------------------|------------|
| Number of hidden layers | 2          |
| Activation Function     | Leaky_ReLU |
| Optimizer               | SGD        |

It achieves the following results

| Attribute               | Value      |
|-------------------------|------------|
| Train Accuracy          | 99.74%     |
| Validation Accuracy     | 97.82%     |
| Test Accuracy           | 97.73%     |


### Basic MLP with Batch Norm - basic_mlp_batch_norm.ipynb
In this notebook we take the basic MLP from the first notebook, but now add a normalization function before the activation function in the hidden layers.
I will use batch norm as my normalization function

Model setup:

| Attribute               | Value     |
|-------------------------|-----------|
| Number of hidden layers | 2         |
| Activation Function     | ReLU      |
| Optimizer               | SGD       |
| Normalization Function  | BatchNorm |

It achieves the following results:

| Attribute           | Value   |
|---------------------|---------|
| Train Accuracy      | 100.00% |
| Validation Accuracy | 98.13%  |
| Test Accuracy       | 97.86%  |


### Basic MLP with AdamW - basic_mlp_adamw.ipynb
This notebook has the same MLP as the basic MLP notebook, the only difference is that I use AdamW as my optimizer

Model setup:

| Attribute               | Value     |
|-------------------------|-----------|
| Number of hidden layers | 2         |
| Activation Function     | ReLU      |
| Optimizer               | AdamW     |

It achieves the following results:

| Attribute           | Value  |
|---------------------|--------|
| Train Accuracy      | 99.70% |
| Validation Accuracy | 97.75% |
| Test Accuracy       | 97.37% |

### Random Experiments
This notebook has different model configurations that I am experimenting with.


