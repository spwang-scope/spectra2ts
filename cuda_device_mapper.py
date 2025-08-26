import torch
import subprocess
import re
import os
from typing import Dict, Optional

class CUDADeviceMapper:
    def __init__(self):
        self.nvidia_to_pytorch_map = self._create_device_mapping()
        self.pytorch_to_nvidia_map = {v: k for k, v in self.nvidia_to_pytorch_map.items()}
    
    def _get_nvidia_gpu_info(self) -> Dict[int, str]:
        """Get GPU information from nvidia-smi"""
        try:
            # Get detailed GPU info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,uuid', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            gpu_info = {}
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        gpu_id = int(parts[0])
                        gpu_name = parts[1]
                        gpu_info[gpu_id] = gpu_name
            
            return gpu_info
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: nvidia-smi not available. Falling back to PyTorch enumeration.")
            return {}
    
    def _create_device_mapping(self) -> Dict[int, int]:
        """Create mapping from nvidia-smi GPU ID to PyTorch device index"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        nvidia_gpu_info = self._get_nvidia_gpu_info()
        if not nvidia_gpu_info:
            # Fallback: assume 1:1 mapping
            return {i: i for i in range(torch.cuda.device_count())}
        
        mapping = {}
        pytorch_device_count = torch.cuda.device_count()
        
        # Get PyTorch device names
        pytorch_devices = {}
        for i in range(pytorch_device_count):
            props = torch.cuda.get_device_properties(i)
            pytorch_devices[i] = props.name
        
        # Create mapping by matching device names
        used_pytorch_devices = set()
        
        for nvidia_id in sorted(nvidia_gpu_info.keys()):
            nvidia_name = nvidia_gpu_info[nvidia_id]
            
            # Find matching PyTorch device
            for pytorch_id, pytorch_name in pytorch_devices.items():
                if pytorch_id in used_pytorch_devices:
                    continue
                
                # Match device names (handle slight variations)
                if self._names_match(nvidia_name, pytorch_name):
                    mapping[nvidia_id] = pytorch_id
                    used_pytorch_devices.add(pytorch_id)
                    break
        
        return mapping
    
    def _names_match(self, name1: str, name2: str) -> bool:
        """Check if two GPU names match (handling minor variations)"""
        # Normalize names for comparison
        name1 = name1.lower().replace(' ', '').replace('-', '')
        name2 = name2.lower().replace(' ', '').replace('-', '')
        return name1 == name2
    
    def get_pytorch_device_from_nvidia_id(self, nvidia_gpu_id: int) -> Optional[int]:
        """Get PyTorch device index from nvidia-smi GPU ID"""
        return self.nvidia_to_pytorch_map.get(nvidia_gpu_id)
    
    def get_nvidia_id_from_pytorch_device(self, pytorch_device: int) -> Optional[int]:
        """Get nvidia-smi GPU ID from PyTorch device index"""
        return self.pytorch_to_nvidia_map.get(pytorch_device)
    
    def set_device_from_config(self, config_device: int) -> torch.device:
        """
        Set PyTorch device based on config string that refers to nvidia-smi GPU IDs
        
        Args:
            config_device: String like "cuda:0", "cuda:1", etc. where the number 
                          refers to nvidia-smi GPU ID (not PyTorch device index)
        
        Returns:
            torch.device: The actual PyTorch device object
        """
        
        # Extract nvidia GPU ID from config
        nvidia_gpu_id = int(config_device)
        
        # Get corresponding PyTorch device index
        pytorch_device_idx = self.get_pytorch_device_from_nvidia_id(nvidia_gpu_id)
        
        if pytorch_device_idx is None:
            available_ids = list(self.nvidia_to_pytorch_map.keys())
            raise ValueError(
                f"nvidia-smi GPU {nvidia_gpu_id} not found or not available to PyTorch. "
                f"Available nvidia-smi GPU IDs: {available_ids}"
            )
        
        # Set the PyTorch device
        pytorch_device = torch.device(f'cuda:{pytorch_device_idx}')
        torch.cuda.set_device(pytorch_device)
        
        print(f"Config requested nvidia-smi GPU {nvidia_gpu_id} -> "
              f"Using PyTorch device {pytorch_device_idx}")
        
        return pytorch_device
    
    def print_mapping(self):
        """Print the mapping between nvidia-smi GPU IDs and PyTorch device indices"""
        print("GPU Device Mapping:")
        print("nvidia-smi GPU ID -> PyTorch Device Index")
        print("-" * 45)
        
        for nvidia_id in sorted(self.nvidia_to_pytorch_map.keys()):
            pytorch_idx = self.nvidia_to_pytorch_map[nvidia_id]
            props = torch.cuda.get_device_properties(pytorch_idx)
            print(f"GPU {nvidia_id:2d} -> PyTorch Device {pytorch_idx} ({props.name})")


# # Usage example
# def main():
#     # Initialize the mapper
#     mapper = CUDADeviceMapper()
    
#     # Print current mapping
#     mapper.print_mapping()
#     print()
    
#     # Example: Your config says "cuda:0" (referring to nvidia-smi GPU 0)
#     config_device = "cuda:7"  # This means nvidia-smi GPU 0 (RTX A6000)
    
#     try:
#         # Set device based on config
#         device = mapper.set_device_from_config(config_device)
        
#         # Test with a simple tensor operation
#         test_tensor = torch.randn(3, 3).to(device)
#         print(f"Successfully created tensor on {device}")
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"Device name: {torch.cuda.get_device_name(device)}")
        
#     except ValueError as e:
#         print(f"Error: {e}")


# if __name__ == "__main__":
#     main()