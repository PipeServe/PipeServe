import json
from pathlib import Path

GPU_CONFIG_FILE = "gpu_config.json"


class GPU:

    def __init__(self, name, flops_efficiency, hbm_memory_efficiency,
                 mem_per_GPU_in_GB):
        self.name = name
        self.flops_efficiency = flops_efficiency
        self.hbm_memory_efficiency = hbm_memory_efficiency
        self.mem_per_GPU_in_GB = mem_per_GPU_in_GB


def load_gpus_from_config(gpu_name):
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