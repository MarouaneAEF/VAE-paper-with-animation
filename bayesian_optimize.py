#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization for VAE model using Optuna.
"""
import os
import argparse
import torch
import numpy as np
import optuna
from optuna.trial import Trial
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from tqdm import tqdm
import random
from datetime import datetime

# Import your existing modules
from custom_dataset import create_image_datasets
from vae_model import VAE

def parse_args():
    parser = argparse.ArgumentParser(description='Optimize VAE hyperparameters')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--img_size', type=int, default=128, help='Image size for resizing')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optimization trials')
    parser.add_argument('--subset_size', type=int, default=100, help='Images to use for optimization')
    parser.add_argument('--quick_epochs', type=int, default=5, help='Epochs for each trial')
    parser.add_argument('--device', type=str, default='mps', help='Device: cuda, mps, or cpu')
    return parser.parse_args()

def objective(trial: Trial, data_loader, device, img_size, quick_epochs):
    """Objective function for Optuna to minimize"""
    
    # Sample hyperparameters
    beta = trial.suggest_float('beta', 0.01, 1.0, log=True)
    latent_dim = trial.suggest_int('latent_dim', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Create model and move to device
    model = VAE(in_channels=3, latent_dim=latent_dim, input_size=img_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Quick training loop
    model.train()
    losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(quick_epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        num_batches = 0
        
        for batch, _ in data_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, log_var = model(batch)
            
            # Get individual loss components
            loss, recon_loss, kl_loss = model.loss_function(
                recon_batch, batch, mu, log_var, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            num_batches += 1
        
        # Average losses for this epoch
        avg_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_kl_loss = epoch_kl_loss / num_batches
        
        # Store all losses
        losses.append(avg_loss)
        recon_losses.append(avg_recon_loss)
        kl_losses.append(avg_kl_loss)
        
        # Print detailed loss information
        print(f"\nTrial {trial.number}, Epoch {epoch+1}/{quick_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Reconstruction Loss: {avg_recon_loss:.4f}")
        print(f"  KL Loss: {avg_kl_loss:.4f}")
        print(f"  Beta: {beta:.4f}")
        
        # Report to Optuna for pruning
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # Store the best losses in trial user attributes
    trial.set_user_attr('best_recon_loss', min(recon_losses))
    trial.set_user_attr('best_kl_loss', min(kl_losses))
    
    return min(losses)

def main():
    args = parse_args()
    
    # Set device with better messaging
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("WARNING: Using CPU - No GPU available")
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    print(f"Loading images from {args.img_dir}...")
    train_dataset, _ = create_image_datasets(
        args.img_dir, 
        transform=transform,
        target_size=(args.img_size, args.img_size),
        quiet=True
    )
    
    # Create a subset for faster optimization
    subset_size = min(args.subset_size, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:subset_size].tolist()
    subset_dataset = Subset(train_dataset, indices)
    
    # Create data loader
    data_loader = DataLoader(
        subset_dataset, 
        batch_size=8,  # Fixed batch size for optimization
        shuffle=True
    )
    
    print(f"Starting optimization with {args.n_trials} trials, using {subset_size} images")
    
    # Create timestamp for study name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"vae_optimization_{timestamp}"
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, data_loader, device, args.img_size, args.quick_epochs
        ),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n=== Optimization Results ===")
    print(f"Best total loss: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nBest trial details:")
    best_trial = study.best_trial
    print(f"  Reconstruction Loss: {best_trial.user_attrs['best_recon_loss']:.4f}")
    print(f"  KL Loss: {best_trial.user_attrs['best_kl_loss']:.4f}")
    
    # Generate training command
    command = f"""
python run_personal_training.py \\
  --img_dir {args.img_dir} \\
  --img_size {args.img_size} \\
  --beta {study.best_params['beta']:.4f} \\
  --latent_dim {study.best_params['latent_dim']} \\
  --device {args.device}
"""
    
    # Save more detailed results
    results_file = f"vae_best_params_{timestamp}.txt"
    with open(results_file, "w") as f:
        f.write("=== VAE Optimization Results ===\n")
        f.write(f"Best total loss: {study.best_value:.4f}\n")
        f.write(f"Best reconstruction loss: {best_trial.user_attrs['best_recon_loss']:.4f}\n")
        f.write(f"Best KL loss: {best_trial.user_attrs['best_kl_loss']:.4f}\n")
        f.write("\nBest hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nRecommended training command:\n")
        f.write(command)
    
    print(f"\nDetailed results saved to {results_file}")
    print("\nRecommended training command:")
    print(command)

if __name__ == "__main__":
    main()