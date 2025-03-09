import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    The Encoder module of the Variational Autoencoder.
    
    It maps the input images to a distribution in the latent space,
    parameterized by mean (mu) and log variance (log_var).
    """
    def __init__(self, in_channels, latent_dim):
        super(Encoder, self).__init__()
        
        # CIFAR-10 images are 32x32, so we design a CNN architecture accordingly
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # 8x8 -> 4x4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         # 4x4 -> 2x2
        
        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Calculate the size of the flattened features
        self.flat_dim = 256 * 2 * 2  # 256 channels x 2x2 spatial dims
        
        # Linear layers for mu and log_var
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
        """
        # Convolutional layers with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Flatten the tensor
        x = x.view(-1, self.flat_dim)
        
        # Get mu and log_var
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    The Decoder module of the Variational Autoencoder.
    
    It maps points from the latent space back to the image space.
    """
    def __init__(self, latent_dim, out_channels):
        super(Decoder, self).__init__()
        
        # Initial projection from latent space
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Transposed convolution layers (also known as deconvolution)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 2x2 -> 4x4
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 4x4 -> 8x8
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 8x8 -> 16x16
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 16x16 -> 32x32
        
        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, z):
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector of shape [batch_size, latent_dim]
        
        Returns:
            x: Reconstructed image of shape [batch_size, channels, height, width]
        """
        # Project and reshape
        x = self.fc(z)
        x = x.view(-1, 256, 2, 2)
        
        # Transposed convolutions with ReLU activations and batch normalization
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        
        # Final layer with sigmoid activation
        # For CIFAR, which has pixel values in [0, 1], sigmoid ensures the output is in the same range
        x = torch.sigmoid(self.deconv4(x))
        
        return x


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    
    Consists of an Encoder that maps inputs to a latent distribution
    and a Decoder that reconstructs inputs from the latent space.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Create encoder and decoder
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels)
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: sample from a Gaussian with given parameters.
        
        This makes the model differentiable despite the sampling step.
        
        Args:
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # sample noise from a standard normal distribution
        z = mu + eps * std  # reparameterization trick
        return z
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
        
        Returns:
            recon_x: Reconstructed input
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
        """
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize (sample from the latent distribution)
        z = self.reparameterize(mu, log_var)
        
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, log_var
    
    def loss_function(self, recon_x, x, mu, log_var, beta=1.0):
        """
        VAE loss function: reconstruction loss + KL divergence.
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of the latent Gaussian
            log_var: Log variance of the latent Gaussian
            beta: Weight of the KL divergence term (beta-VAE)
        
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss: binary cross entropy for images in [0, 1]
        # We use sum instead of mean to be consistent with the KL formulation
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence: closed form for Gaussian, formula: 0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss with the beta parameter
        total_loss = recon_loss + beta * kl_loss
        
        # Normalize by batch size for stable training
        batch_size = x.size(0)
        total_loss /= batch_size
        recon_loss /= batch_size
        kl_loss /= batch_size
        
        return total_loss, recon_loss, kl_loss
    
    def generate(self, num_samples, device):
        """
        Generate new images by sampling from the latent space.
        
        Args:
            num_samples: Number of images to generate
            device: Device to use (cuda/cpu)
        
        Returns:
            generated_images: Generated image tensors
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(device)
        
        # Decode
        with torch.no_grad():
            generated_images = self.decoder(z)
        
        return generated_images 