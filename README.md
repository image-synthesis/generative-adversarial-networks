## Neural Style Transfer

## Introduction

This repository contains a pytorch implementation of WGAN with gradient penalty.

## Installation and Setup

1. Clone the repository:
    ```
    git clone https://github.com/image-synthesis/generative-adversarial-networks.git
    cd generative-adversarial-networks
    ```

2. Create a conda environment:
    ```
    conda create --name gan_env python=3.10
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    pip install -e .
    ```

### Training Parameters

This project uses a YAML configuration file to manage the training parameters for the GAN. A sample config file for mnist can be found in ```src/config/mnist_config.yml```.

# Configuration Overview

This section provides a detailed explanation of the configurations used for training the Generative Adversarial Network (GAN), including settings for the generator, discriminator, and trainer. Note that the number of individual layers in either the generator or discriminator can be modified.

## Generator Configuration
The generator is responsible for producing synthetic images from random noise vectors.

- **`latent_dim`**: The dimensionality of the input noise vector. Default: `100`
- **`fc_layer`**:
  - `output_channels`: Number of output channels after the fully connected layer. Default: `128`
  - `output_height`: Height of the output feature map. Default: `7`
  - `output_width`: Width of the output feature map. Default: `7`

- **`deconv_layer_params`**:
  - **Layer 1**:
    - `input_channels`: Number of input channels from the fully connected layer. Default: `128`
    - `output_channels`: Number of output channels produced. Default: `64`
    - `kernel_size`: Size of the deconvolution kernel. Default: `4`
    - `stride`: Stride of the deconvolution. Default: `2`
    - `padding`: Padding added to the input feature map. Default: `1`
  
  - **Layer 2**:
    - `input_channels`: Number of input channels from the previous layer. Default: `64`
    - `output_channels`: Number of output channels produced (grayscale image). Default: `1`
    - `kernel_size`: Size of the deconvolution kernel. Default: `4`
    - `stride`: Stride of the deconvolution. Default: `2`
    - `padding`: Padding added to the input feature map. Default: `1`

## Discriminator Configuration
The discriminator differentiates between real and generated images.

- **`conv_layer_params`**:
  - **Layer 1**:
    - `input_channels`: Number of input channels from the image. Default: `1`
    - `output_channels`: Number of output channels produced. Default: `64`
    - `kernel_size`: Size of the convolutional kernel. Default: `4`
    - `stride`: Stride of the convolution. Default: `2`
    - `padding`: Padding added to the input feature map. Default: `1`
  
  - **Layer 2**:
    - `input_channels`: Number of input channels from the previous layer. Default: `64`
    - `output_channels`: Number of output channels produced. Default: `128`
    - `kernel_size`: Size of the convolutional kernel. Default: `4`
    - `stride`: Stride of the convolution. Default: `2`
    - `padding`: Padding added to the input feature map. Default: `1`

- **`fc_layer`**:
  - `input_channels`: Number of flattened input channels from the last convolutional layer. Default: `6272`

## Trainer Configuration
The trainer handles the GAN training process.

- **`output_dir`**: Directory where training outputs and logs will be saved. Example: `/home/ubuntu/generative-adversarial-networks/src/output`
- **`experiment`**: Name of the training experiment. Example: `mnist`
- **`k`**: Number of discriminator updates per generator update. Default: `5`
- **`batch_size`**: Number of samples per batch during training. Default: `64`

- **Learning Rates (`lrs`)**:
  - `lr_gen`: Learning rate for the generator. Default: `0.0001`
  - `lr_disc`: Learning rate for the discriminator. Default: `0.0001`

- **Betas (`betas`)**:
  - `betas_gen`: Betas for the Adam optimizer of the generator. Default: `[0, 0.9]`
  - `betas_disc`: Betas for the Adam optimizer of the discriminator. Default: `[0, 0.9]`

- **Training Settings**:
  - `iterations`: Number of training iterations. Default: `100000`
  - `log_steps`: Frequency of logging performance metrics. Default: `500`
  - `ckpt_steps`: Frequency of saving model checkpoints. Default: `5000`
  - `ckpt_dir`: Directory where checkpoints are saved. Example: `ckpt`

These configurations ensure proper model initialization, training flow, and output management during GAN training.

## Usage

First create your configuration yaml file using the training params above and create a script which loads your dataset and passes it along with the config to the train function. You can refer to `src/mnist_gan.py` for an example.

```
python <path to main gan script> --config <path to config file>
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)