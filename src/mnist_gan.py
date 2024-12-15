"""
This script trains a Generative Adversarial Network (GAN) on the MNIST dataset.

Modules:
    torch: PyTorch library for tensor computations and neural networks.
    torchvision: Library for computer vision tasks, including datasets and transformations.
    yaml: Library for parsing YAML configuration files.
    argparse: Library for parsing command-line arguments.
    utils.train: Custom module containing the training function.

Functions:
    main: The main function that loads the dataset, parses command-line arguments, reads the configuration file, and starts the training process.

Usage:
    python mnist_gan.py --config-file <path_to_config_file>

Arguments:
    --config-file: Path to the YAML configuration file containing training parameters.
"""
import torch
from torchvision import datasets, transforms
from utils.train import *

import yaml
import argparse

if __name__ == '__main__':
    # load the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root='data',
        train=True,
        transform=transform,
        download=True
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)

    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    train(config, train_dataset)
