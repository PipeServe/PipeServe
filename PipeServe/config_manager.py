"""Configuration management for BatchConfigurator with JSON support

This module provides utilities for loading configuration from JSON files
and saving results back to JSON format.
"""

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add llm-analysis to Python path
sys.path.append(str(Path(__file__).parent / "llm-analysis"))

from llm_analysis.analysis import LLMAnalysis
from llm_analysis.config import (
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)

try:
    from .gpu_config import load_gpus_from_config
except ImportError:
    from gpu_config import load_gpus_from_config


@dataclass
class BatchConfigParams:
    """Configuration parameters for BatchConfigurator"""
    
    # Model and hardware configuration
    model: str = "Llama-2-7b"
    gpu_name: str = "t4-pcie-16gb"
    dtype: str = "w16a16e16"
    
    # Sequence and batch configuration
    max_chunk_size: int = 256
    
    # SLO constraints
    slo_prefill: float = 1.5
    slo_decode: float = 0.5
    
    # Parallelism configuration
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    dp_size: int = 1
    
    
    # Algorithm selection
    algorithm: str = "both"  # "brute-force", "bucket", "both"


@dataclass
class BatchOptimizationResult:
    """Results from batch optimization"""
    
    # Best configuration found
    best_batch_p: int
    best_batch_d: int
    best_partition: List[int]
    best_delta: float
    best_cost: float
    
    # Pipeline configuration
    max_stage: int
    max_layer_per_gpu: int
    
    # Latency measurements
    prefill_latency: float
    decode_latency: float
    total_prefill_latency: float  # Including pipeline overhead
    total_decode_latency: float   # Including pipeline overhead
    
    # Performance metrics
    dfs_count: int = 0
    dfs_original_count: int = 0
    solve_time: float = 0.0
    
    # SLO compliance
    meets_slo: bool = True
    
    # Configuration used
    config_params: Optional[Dict[str, Any]] = None


class ConfigManager:
    """Manager for handling JSON configuration loading and result saving"""
    
    MODEL_CONFIG_FILE = str(Path(__file__).parent / "model_configs")
    
    @classmethod
    def load_config_from_json(cls, json_file_path: str) -> BatchConfigParams:
        """Load configuration from JSON file
        
        Args:
            json_file_path (str): Path to the JSON configuration file
            
        Returns:
            BatchConfigParams: Configuration parameters loaded from JSON
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return BatchConfigParams(**config_dict)
    
    @classmethod
    def save_result_to_json(cls, result: BatchOptimizationResult, json_file_path: str) -> None:
        """Save optimization result to JSON file
        
        Args:
            result (BatchOptimizationResult): Optimization result to save
            json_file_path (str): Path to save the JSON result file
        """
        result_dict = asdict(result)
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def create_analysis_from_config(cls, config: BatchConfigParams) -> LLMAnalysis:
        """Create LLMAnalysis instance from configuration
        
        Args:
            config (BatchConfigParams): Configuration parameters
            
        Returns:
            LLMAnalysis: Configured analysis instance
        """
        # Get full model path
        model_name = str(Path(cls.MODEL_CONFIG_FILE) / config.model)
        
        # Load configurations
        model_config = get_model_config_by_name(model_name)
        gpu_config = get_gpu_config_by_name(config.gpu_name)
        dtype_config = get_dtype_config_by_name(config.dtype)
        parallel_config = ParallelismConfig(
            tp_size=config.tp_size,
            pp_size=config.pp_size,
            sp_size=config.sp_size,
            dp_size=config.dp_size,
        )
        
        # Load GPU efficiency parameters
        gpu = load_gpus_from_config(config.gpu_name)
        
        # Create analysis instance
        analysis = LLMAnalysis(
            model_config,
            gpu_config,
            dtype_config,
            parallel_config,
            flops_efficiency=gpu.flops_efficiency,
            hbm_memory_efficiency=gpu.hbm_memory_efficiency,
        )
        
        return analysis
    
    @classmethod
    def create_sample_config(cls, output_path: str) -> None:
        """Create a sample configuration JSON file
        
        Args:
            output_path (str): Path to save the sample configuration
        """
        sample_config = BatchConfigParams()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(sample_config), f, indent=2, ensure_ascii=False)
        
        print(f"Sample configuration saved to: {output_path}")
