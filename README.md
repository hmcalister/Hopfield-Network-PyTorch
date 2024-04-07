# Hopfield Network PyTorch

An implementation of the Hopfield Network using PyTorch for the linear algebra calculations. Namely, this implementation is looking to leverage the power of the GPU using PyTorch's CUDA backend for linear algebra operations.

This package provides a consistent API to both the traditional and modern Hopfield networks for autoassociative memory tasks. Using CUDA allows even large memory tasks to be trained and tested quickly using the GPU. 

## Installation

Clone this repository, then `cd` into the project root directory. Run `pip install .` to install the package, or `pip install -e .` to install as editable (if you plan to make changes to the package).

Use the package in any python script by importing `HopfieldNetworkPyTorch.HopfieldNetwork` or `HopfieldNetworkPyTorch.ModernHopfieldNetwork`. See the `examples` directory for worked examples.

## Acknowledgments

The modern Hopfield network implementation is based on the mathematics provided in [Dense Associative Memory for Pattern Recognition (Krotov and Hopfield, 2016)](https://arxiv.org/abs/1606.01164), as well as referencing [Krotov's implementation for a classification task](https://github.com/DimaKrotov/Dense_Associative_Memory/blob/master/Dense_Associative_Memory_training.ipynb).