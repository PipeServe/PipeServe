import json
from pathlib import Path

GPU_CONFIG_FILE = "gpu_config.json"


class GPU:
    """GPU configuration class that stores GPU specifications and performance metrics"""

    def __init__(self, name, flops_efficiency, hbm_memory_efficiency,
                 mem_per_GPU_in_GB):
        """Initialize GPU configuration
        
        Args:
            name (str): Name of the GPU model
            flops_efficiency (float): FLOPS efficiency ratio (0.0 to 1.0)
            hbm_memory_efficiency (float): High bandwidth memory efficiency ratio (0.0 to 1.0)
            mem_per_GPU_in_GB (int): Memory capacity per GPU in gigabytes
        """
        self.name = name
        self.flops_efficiency = flops_efficiency
        self.hbm_memory_efficiency = hbm_memory_efficiency
        self.mem_per_GPU_in_GB = mem_per_GPU_in_GB


def load_gpus_from_config(gpu_name):
    """Load GPU configuration from JSON configuration file
    
    Args:
        gpu_name (str): Name of the GPU to load configuration for
        
    Returns:
        GPU: GPU object with loaded configuration parameters
        
    Raises:
        FileNotFoundError: If GPU configuration file is not found
        KeyError: If specified GPU name is not found in configuration
    """
    with open(f"{Path(__file__).parent}/{GPU_CONFIG_FILE}", 'r') as file:
        data = json.load(file)
        # gpus = []
        for name, attributes in data.items():
            if name == gpu_name:
                gpu = GPU(
                    name,
                    flops_efficiency=attributes['flops_efficiency'],
                    hbm_memory_efficiency=attributes['hbm_memory_efficiency'],
                    mem_per_GPU_in_GB=attributes['mem_per_GPU_in_GB'])
        return gpu