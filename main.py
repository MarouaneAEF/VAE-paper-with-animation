import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys

from vae_model import VAE
from utils import (
    plot_reconstructed_images, 
    plot_generated_images, 
    plot_latent_space, 
    create_reconstruction_video,
    close_video_writer
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a VAE on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta coefficient for KL divergence term')
    parser.add_argument('--save_path', type=str, default='vae_cifar10.pth', help='Path to save model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--reconstruction_interval', type=int, default=1, 
                        help='Save reconstruction images every N epochs')
    parser.add_argument('--video_fps', type=int, default=2, 
                        help='Frames per second for the reconstruction progress video')
    parser.add_argument('--quick', action='store_true',
                        help='Run a quicker version with smaller dataset portion')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    
    for data, _ in tqdm(dataloader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, log_var = model(data)
        
        # Calculate loss
        loss, recon, kl = model.loss_function(recon_batch, data, mu, log_var, beta)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        recon_loss += recon.item()
        kl_loss += kl.item()
    
    return train_loss / len(dataloader), recon_loss / len(dataloader), kl_loss / len(dataloader)

def validate(model, dataloader, beta, device):
    model.eval()
    val_loss = 0
    recon_loss = 0
    kl_loss = 0
    
    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Validation"):
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, log_var = model(data)
            
            # Calculate loss
            loss, recon, kl = model.loss_function(recon_batch, data, mu, log_var, beta)
            
            val_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()
    
    return val_loss / len(dataloader), recon_loss / len(dataloader), kl_loss / len(dataloader)

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Create reconstruction frames directory
    os.makedirs('reconstruction_frames', exist_ok=True)
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print("\n🔄 Loading CIFAR-10 dataset...")
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # For quick mode, use a subset of the data
    if args.quick:
        print("⚡ Quick mode enabled - using subset of data")
        train_subset_size = 5000  # Use only 5000 training examples
        test_subset_size = 1000   # Use only 1000 test examples
        
        # Create subsets
        from torch.utils.data import Subset
        import random
        
        train_indices = random.sample(range(len(train_dataset)), train_subset_size)
        test_indices = random.sample(range(len(test_dataset)), test_subset_size)
        
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"📊 Training dataset size: {len(train_dataset)}")
    print(f"📊 Test dataset size: {len(test_dataset)}")
    
    print(f"\n🧠 Creating VAE model with latent dimension {args.latent_dim}...")
    model = VAE(
        in_channels=3,
        latent_dim=args.latent_dim
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Get a fixed batch of images for tracking reconstruction progress
    examples, _ = next(iter(test_loader))
    fixed_examples = examples[:8].to(device)  # Take first 8 examples
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"📽️ Video will be updated every {args.reconstruction_interval} epoch(s)")
    print(f"🎬 You can view the video at 'reconstruction_progress.mp4' during training\n")
    
    try:
        # Initial reconstruction before training
        print("📸 Creating initial reconstruction (pre-training)...")
        plot_reconstructed_images(model, fixed_examples, 0, device, update_video=True)
            
        for epoch in range(args.epochs):
            print(f"\n🔄 Epoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, args.beta, device)
            train_losses.append(train_loss)
            
            # Validate
            val_loss, val_recon, val_kl = validate(model, test_loader, args.beta, device)
            val_losses.append(val_loss)
            
            print(f"📉 Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")
            print(f"📊 Valid Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")
            
            # Save reconstruction images at regular intervals and update video
            if (epoch + 1) % args.reconstruction_interval == 0:
                print(f"📸 Saving reconstruction images for epoch {epoch+1}...")
                plot_reconstructed_images(
                    model, 
                    fixed_examples, 
                    epoch+1, 
                    device, 
                    update_video=True
                )
            
            # Show progress 
            progress = (epoch + 1) / args.epochs * 100
            progress_bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
            print(f"\n📊 Training progress: {progress_bar} {progress:.1f}%")
                
        # Save model
        torch.save(model.state_dict(), args.save_path)
        print(f"\n💾 Model saved to {args.save_path}")
        
        # Plot loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_curve.png')
        plt.close()
        print("📈 Loss curve saved to 'loss_curve.png'")
        
        # Final visualizations
        # Sample random test images
        examples, _ = next(iter(test_loader))
        examples = examples[:16].to(device)  # Take 16 examples
        
        # Plot final reconstructions
        print("\n🖼️ Creating final visualizations...")
        plot_reconstructed_images(
            model, examples, args.epochs, device, 
            num_examples=16, filename='final_reconstructions.png', update_video=False
        )
        print("🖼️ Final reconstructions saved to 'final_reconstructions.png'")
        
        # Plot final random samples from the model
        plot_generated_images(model, 16, args.epochs, device, grid_size=(4, 4), filename='final_samples.png')
        print("🖼️ Generated samples saved to 'final_samples.png'")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise
    finally:
        # Make sure we close the video writer properly even if training is interrupted
        close_video_writer()
        print("\n✅ Training complete. The reconstruction video has been saved to 'reconstruction_progress.mp4'.")
        print("🎬 You can view the video to see how reconstructions improved during training.")

if __name__ == "__main__":
    main() 