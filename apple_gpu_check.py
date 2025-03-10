import psutil
import os
import time
import torch
import platform

def get_apple_silicon_info():
    """Get information about the Apple Silicon system"""
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Get CPU information
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"\nCPU: {physical_cores} physical cores, {logical_cores} logical cores")
    
    # Get memory information
    memory = psutil.virtual_memory()
    print(f"\nRAM: {memory.total / (1024 ** 3):.2f} GB total")
    print(f"RAM Usage: {memory.percent}%")
    
    # PyTorch GPU information
    print("\nPyTorch GPU Information:")
    print(f"PyTorch version: {torch.__version__}")
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
        print("PyTorch can use Apple GPU for acceleration")
        
        # Create a test tensor on GPU to check if it works
        try:
            device = torch.device("mps")
            x = torch.rand(1000, 1000).to(device)
            print(f"Successfully created a {x.shape} tensor on Apple GPU")
            print(f"Tensor device: {x.device}")
            
            # Basic benchmark
            print("\nRunning a quick matrix multiplication benchmark...")
            start_time = time.time()
            result = torch.matmul(x, x)
            torch.mps.synchronize()  # Wait for GPU operations to complete
            end_time = time.time()
            print(f"Time: {end_time - start_time:.4f} seconds")
        except Exception as e:
            print(f"Error when trying to use MPS: {e}")
    else:
        print("MPS is not available. PyTorch will use CPU only.")
        
    print("\nFor full GPU monitoring on Apple Silicon:")
    print("1. Use Activity Monitor app (GUI)")
    print("2. Run 'sudo powermetrics --samplers gpu_power' in terminal")
    print("3. Run 'sudo asitop' in terminal")
        
if __name__ == "__main__":
    get_apple_silicon_info() 