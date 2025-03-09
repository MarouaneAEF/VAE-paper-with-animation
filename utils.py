import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import imageio
import glob
import time

# Global variable to store video writer
video_writer = None
VIDEO_FILENAME = 'reconstruction_progress.mp4'

def plot_reconstructed_images(model, original_images, epoch, device, num_examples=8, filename=None, update_video=True):
    """
    Plot original images and their reconstructions side by side.
    
    Args:
        model: VAE model
        original_images: Batch of original images
        epoch: Current epoch number (for filename)
        device: Device to use
        num_examples: Number of examples to display
        filename: If provided, save to this file instead of showing
        update_video: Whether to update the video with the new frame
    """
    global video_writer, VIDEO_FILENAME
    
    # Take only the specified number of examples
    if original_images.size(0) > num_examples:
        original_images = original_images[:num_examples]
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, _, _ = model(original_images)
    
    # Convert to numpy arrays for plotting (move to CPU first)
    original_images = original_images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2, 4))
    
    for i in range(num_examples):
        # Original images (transpose to have channels last as required by imshow)
        axes[0, i].imshow(np.transpose(original_images[i], (1, 2, 0)))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed images
        axes[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Add epoch information
    fig.suptitle(f'Epoch {epoch}', fontsize=16)
    
    # Save frame
    save_dir = 'reconstruction_frames'
    os.makedirs(save_dir, exist_ok=True)
    frame_path = f'{save_dir}/reconstruction_epoch_{epoch:03d}.png'
    
    # Save or display
    if filename is None:
        plt.savefig(frame_path)
    else:
        plt.savefig(filename)
    
    plt.close()
    
    # Update video in real-time if requested
    if update_video:
        update_reconstruction_video(frame_path)
        print(f"✅ Frame {epoch} added to video - You can view '{VIDEO_FILENAME}' while training continues")


def update_reconstruction_video(new_frame_path, fps=4):
    """
    Update the reconstruction video with a new frame.
    
    Args:
        new_frame_path: Path to the new frame image
        fps: Frames per second for the video
    """
    global video_writer, VIDEO_FILENAME
    
    try:
        # Read the new frame
        new_frame = imageio.imread(new_frame_path)
        
        # Initialize or update the video writer
        if video_writer is None:
            print(f"\n🎬 Creating reconstruction video: '{VIDEO_FILENAME}'")
            # Use FFMPEG with explicitly specified format and codec
            try:
                # First try with h264 codec
                video_writer = imageio.get_writer(
                    VIDEO_FILENAME,
                    format='FFMPEG',
                    mode='I',
                    fps=fps,
                    codec='libx264',
                    pixelformat='yuv420p',
                    quality=8
                )
            except Exception as e:
                print(f"Falling back to alternative method due to: {e}")
                # If that fails, try to create GIF instead
                if VIDEO_FILENAME.endswith('.mp4'):
                    VIDEO_FILENAME = VIDEO_FILENAME.replace('.mp4', '.gif')
                    print(f"Switching to GIF format: {VIDEO_FILENAME}")
                
                video_writer = imageio.get_writer(
                    VIDEO_FILENAME,
                    format='GIF',
                    mode='I',
                    fps=fps
                )
        
        # Add the new frame to the video
        video_writer.append_data(new_frame)
        
    except Exception as e:
        print(f"Warning: Error updating video: {e}")
        # Create a new writer if there was an error
        close_video_writer()


def close_video_writer():
    """Close the video writer to ensure the video file is properly saved."""
    global video_writer, VIDEO_FILENAME
    if video_writer is not None:
        try:
            video_writer.close()
            video_writer = None
            print(f"\n🎥 Video saved successfully: '{VIDEO_FILENAME}'")
        except Exception as e:
            print(f"Error closing video writer: {e}")


def create_reconstruction_video(output_filename='reconstruction_progress.mp4', fps=4):
    """
    Create a video from the reconstruction images saved during training.
    This function can be used to recreate the video from saved frames.
    
    Args:
        output_filename: Name of the output video file
        fps: Frames per second for the video
    """
    global video_writer, VIDEO_FILENAME
    
    # Update global video filename
    VIDEO_FILENAME = output_filename
    
    # Close the existing video writer if it's open
    close_video_writer()
    
    # Create the frames directory if it doesn't exist
    frames_dir = 'reconstruction_frames'
    if not os.path.exists(frames_dir):
        print(f"No reconstruction frames found in {frames_dir}")
        return
    
    # Get all the frame files and sort them
    frame_files = sorted(glob.glob(f'{frames_dir}/reconstruction_epoch_*.png'))
    
    if len(frame_files) == 0:
        print(f"No reconstruction frames found in {frames_dir}")
        return
    
    print(f"Creating video from {len(frame_files)} frames...")
    
    # Create GIF instead of MP4 if we had issues with the codec
    if output_filename.endswith('.mp4'):
        try:
            # Try to create MP4 first
            video_writer = imageio.get_writer(
                output_filename, 
                format='FFMPEG',
                mode='I',
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=8
            )
        except Exception as e:
            print(f"Error creating MP4 writer: {e}. Falling back to GIF format.")
            output_filename = output_filename.replace('.mp4', '.gif')
            VIDEO_FILENAME = output_filename
            video_writer = imageio.get_writer(
                output_filename,
                format='GIF',
                mode='I',
                fps=fps
            )
    else:
        # Use GIF format
        video_writer = imageio.get_writer(
            output_filename,
            format='GIF',
            mode='I',
            fps=fps
        )
    
    # Add all frames
    for i, file in enumerate(frame_files):
        print(f"Adding frame {i+1}/{len(frame_files)}: {file}")
        video_writer.append_data(imageio.imread(file))
    
    # Close and save
    close_video_writer()
    
    print(f"Video saved to {VIDEO_FILENAME}")


def plot_generated_images(model, num_samples, epoch, device, grid_size=(2, 4), filename=None):
    """
    Generate and plot random samples from the model.
    
    Args:
        model: VAE model
        num_samples: Number of samples to generate
        epoch: Current epoch number (for filename)
        device: Device to use
        grid_size: Tuple of (rows, cols) for the grid layout
        filename: If provided, save to this file instead of showing
    """
    # Generate samples
    generated = model.generate(num_samples, device)
    
    # Convert to numpy array (move to CPU first)
    generated = generated.cpu()
    
    # Create a grid of images
    img_grid = make_grid(generated, nrow=grid_size[1], normalize=True)
    
    # Plot
    plt.figure(figsize=(grid_size[1] * 2, grid_size[0] * 2))
    plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
    plt.title(f'Generated Samples at Epoch {epoch}')
    plt.axis('off')
    
    # Save or display
    if filename is None:
        plt.savefig(f'generated_epoch_{epoch}.png')
    else:
        plt.savefig(filename)
    
    plt.close()


def plot_latent_space(model, dataloader, device, n_samples=1000, tsne=True, filename=None):
    """
    Visualize the latent space using PCA or t-SNE.
    
    Args:
        model: VAE model
        dataloader: DataLoader for the dataset
        device: Device to use
        n_samples: Maximum number of samples to use for visualization
        tsne: If True, use t-SNE for visualization, otherwise use PCA
        filename: If provided, save to this file instead of showing
    """
    # Collect samples from the dataset
    model.eval()
    latent_vectors = []
    labels = []
    count = 0
    
    with torch.no_grad():
        for data, label in dataloader:
            if count >= n_samples:
                break
            
            # Get batch size
            batch_size = data.size(0)
            
            # Get latent vectors
            data = data.to(device)
            mu, _ = model.encoder(data)
            
            # Move back to CPU and convert to numpy
            latent_vectors.append(mu.cpu().numpy())
            labels.append(label.numpy())
            
            count += batch_size
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:n_samples]
    labels = np.concatenate(labels, axis=0)[:n_samples]
    
    # Reduce dimensions for visualization
    if tsne and model.latent_dim > 2:
        # Use t-SNE for high-dimensional latent spaces
        reducer = TSNE(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(latent_vectors)
        title = 't-SNE Visualization of Latent Space'
    elif model.latent_dim > 2:
        # Use PCA for high-dimensional latent spaces
        reducer = PCA(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(latent_vectors)
        title = 'PCA Visualization of Latent Space'
    else:
        # If latent_dim is already 2, no need to reduce
        reduced_data = latent_vectors
        title = 'Visualization of 2D Latent Space'
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                          c=labels, cmap='tab10', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Class')
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    # Save or display
    if filename is None:
        plt.savefig('latent_space.png')
    else:
        plt.savefig(filename)
    
    plt.close()


def interpolate_latent_space(model, img1, img2, device, steps=10, filename=None):
    """
    Interpolate between two images in the latent space.
    
    Args:
        model: VAE model
        img1: First image tensor
        img2: Second image tensor
        device: Device to use
        steps: Number of interpolation steps
        filename: If provided, save to this file instead of showing
    """
    model.eval()
    
    # Move images to device
    img1 = img1.unsqueeze(0).to(device)  # Add batch dimension
    img2 = img2.unsqueeze(0).to(device)
    
    # Encode images to get latent vectors (mu only)
    with torch.no_grad():
        mu1, _ = model.encoder(img1)
        mu2, _ = model.encoder(img2)
    
    # Interpolate in the latent space
    ratios = np.linspace(0, 1, steps)
    interpolated_images = []
    
    with torch.no_grad():
        for ratio in ratios:
            # Linear interpolation
            mu_interp = mu1 * (1 - ratio) + mu2 * ratio
            
            # Decode
            img_interp = model.decoder(mu_interp)
            interpolated_images.append(img_interp.cpu())
    
    # Concatenate all images
    interpolated_images = torch.cat(interpolated_images, dim=0)
    
    # Create a grid
    img_grid = make_grid(interpolated_images, nrow=steps, normalize=True)
    
    # Plot
    plt.figure(figsize=(15, 3))
    plt.imshow(np.transpose(img_grid.numpy(), (1, 2, 0)))
    plt.title('Latent Space Interpolation')
    plt.axis('off')
    
    # Save or display
    if filename is None:
        plt.savefig('interpolation.png')
    else:
        plt.savefig(filename)
    
    plt.close() 