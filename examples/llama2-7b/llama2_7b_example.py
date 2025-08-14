#!/usr/bin/env python3
"""
Llama2-7B Optimization Example

This example demonstrates how to use the BatchConfigurator to optimize
batch sizes and model partitioning for Llama2-7B model with SLO constraints.

Usage:
    python example.py

Requirements:
    - PipeServe package with all dependencies installed
    - Model configuration files in the appropriate directories
"""

import os
import sys
from pathlib import Path

# Add PipeServe to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "PipeServe"))
sys.path.append(str(Path(__file__).parent))

from BatchConfigurator import BatchConfigurator


def main():
    """Main function to run Llama2-7B optimization example"""
    
    print("=" * 60)
    print("Llama2-7B Batch Configuration Optimization Example")
    print("=" * 60)
    
    # Configuration file path
    script_dir = os.path.dirname(__file__)
    config_file = os.path.join(script_dir, 'config.json')
    result_file = os.path.join(script_dir, 'result.json')
    
    print(f"Loading configuration from: {config_file}")
    
    try:
        # Create BatchConfigurator from JSON configuration
        configurator = BatchConfigurator.from_json_config(config_file)
        
        print("\nConfiguration loaded successfully!")
        print(f"Model: {configurator.analysis.model_config.name}")
        print(f"GPU: {configurator.analysis.gpu_config.name}")
        print(f"SLO Prefill: {configurator.slo_p}s")
        print(f"SLO Decode: {configurator.slo_d}s")
        print(f"Max Chunk Size: {configurator.max_chunk_size}")
        
        print("\nStarting optimization...")
        
        # Run optimization and save results to JSON
        result = configurator.optimize_to_json(result_file)
        
        if result:
            print(f"\nOptimization completed! Results saved to: {result_file}")
            print("\nOptimization Results:")
            print(f"Best Prefill Batch Size: {result.best_batch_p}")
            print(f"Best Decode Batch Size: {result.best_batch_d}")
            print(f"Model Partition: {result.best_partition}")
            print(f"Pipeline Stages: {result.max_stage}")
            print(f"Max Layers per GPU: {result.max_layer_per_gpu}")
            print(f"Prefill Latency: {result.prefill_latency:.3f}s")
            print(f"Decode Latency: {result.decode_latency:.3f}s")
            print(f"Total Prefill Latency: {result.total_prefill_latency:.3f}s")
            print(f"Total Decode Latency: {result.total_decode_latency:.3f}s")
            print(f"Meets SLO: {'Yes' if result.meets_slo else 'No'}")
            print(f"Optimization Time: {result.solve_time:.2f}s")
        else:
            print("\nOptimization failed - no feasible solution found!")
            print("Try adjusting SLO constraints or using different hardware configuration.")
            
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found!")
        print("Make sure you're running this script from the example directory.")
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
