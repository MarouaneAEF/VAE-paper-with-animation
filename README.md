# Variational Autoencoder (VAE) for CIFAR-10

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) trained on the CIFAR-10 dataset. The code is structured to be clear, well-documented, and educational.

## What is a VAE?

A Variational Autoencoder (VAE) is a type of generative model that learns to encode data into a latent space distribution and decode samples from that distribution back into data. Unlike regular autoencoders, VAEs model the latent space as a probability distribution, which allows for generating new data by sampling from this distribution.

VAEs consist of two main components:
1. **Encoder**: Maps input data to parameters (mean and variance) of a probability distribution in latent space.
2. **Decoder**: Maps points from the latent space back to the data space.

The key innovation is the use of a "reparameterization trick" that allows the model to be trained with backpropagation despite the sampling process.

## Theoretical Background

The VAE optimizes two objectives:
1. **Reconstruction Loss**: Ensures the decoded data is similar to the input data.
2. **KL Divergence**: Forces the learned latent distribution to be close to a standard normal distribution.

The overall loss function is:
```
Loss = Reconstruction Loss + β * KL Divergence
```

Where β is a hyperparameter that controls the trade-off between reconstruction quality and the structure of the latent space.

## Implementation Details

Our implementation includes:

### Architecture
- **Encoder**: A series of convolutional layers followed by batch normalization and ReLU activations, with two fully connected layers at the end for the mean and log variance.
- **Decoder**: Mirror of the encoder, with transposed convolutions to upsample from the latent vector back to the input dimensions.

### Training Process
- We use the Adam optimizer with a learning rate of 1e-3.
- The CIFAR-10 dataset is used, which contains 60,000 32x32 color images across 10 classes.
- Training includes monitoring of both the reconstruction loss and KL divergence components.

### Visualization
- Reconstruction comparison between original and reconstructed images
- Random samples generated from the latent space
- 2D visualization of the latent space using t-SNE or PCA
- Interpolation between samples in the latent space

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd vae-cifar10

# Install requirements
pip install -r requirements.txt
```

## Usage

To train the model with default parameters:

```bash
python main.py
```

You can customize training with various parameters:

```bash
python main.py --batch_size 64 --epochs 100 --lr 0.001 --latent_dim 64 --beta 1.0
```

### Parameters

- `--batch_size`: Size of the training batch (default: 128)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--latent_dim`: Dimensionality of the latent space (default: 128)
- `--beta`: Weight of the KL divergence term (default: 1.0)
- `--save_path`: Path to save the trained model (default: 'vae_cifar10.pth')
- `--device`: Device to train on (default: 'cuda' if available, otherwise 'cpu')

## Results

During and after training, the model will generate:

- Loss curves showing training and validation loss over time
- Reconstructed images at regular intervals to show learning progress
- Generated samples from random points in the latent space
- Visualization of the latent space structure

## Files Organization

- `main.py`: Main script for training and evaluating the VAE
- `vae_model.py`: Implementation of the VAE architecture and loss function
- `utils.py`: Utility functions for visualization and data processing
- `requirements.txt`: Required Python packages

## References

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
2. Doersch, C. (2016). Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.

## License

MIT 