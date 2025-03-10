import imageio
import glob
import os
import numpy as np
from PIL import Image

def create_gif_from_frames(frames_dir='reconstruction_frames', output_file='reconstruction_progress.gif', fps=2):
    """
    Create a GIF from PNG frames in the specified directory
    
    Args:
        frames_dir: Directory containing the frame images
        output_file: Output GIF filename
        fps: Frames per second
    """
    # Get all frame files and sort them numerically (based on epoch number)
    frame_files = sorted(glob.glob(f'{frames_dir}/reconstruction_epoch_*.png'))
    
    if len(frame_files) == 0:
        print(f"No frame files found in {frames_dir}")
        return
    
    print(f"Found {len(frame_files)} frame files")
    
    # Read all frames
    frames = []
    for file in frame_files:
        try:
            # Use PIL to open and possibly resize the image
            img = Image.open(file)
            # You can resize here if needed: img = img.resize((width, height))
            frames.append(np.array(img))
            print(f"Added frame: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    # Save as GIF
    try:
        print(f"Creating GIF with {len(frames)} frames at {fps} fps")
        imageio.mimsave(output_file, frames, fps=fps)
        print(f"GIF saved to {output_file}")
    except Exception as e:
        print(f"Error creating GIF: {e}")

if __name__ == "__main__":
    create_gif_from_frames() 