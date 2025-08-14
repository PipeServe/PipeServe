#!/usr/bin/env python3
"""
Llama2-13B Optimization Example

This example demonstrates how to use the BatchConfigurator to optimize
batch sizes and model partitioning for Llama2-13B model with SLO constraints.

The 13B model requires more GPU memory and has different performance
characteristics compared to the 7B model, so we use a more powerful
GPU (RTX6000-48GB) and adjusted SLO constraints.

Usage:
    python example.py

Requirements:
    - PipeServe package with all dependencies installed
    - Model configuration files in the appropriate directories
    - RTX6000 or similar high-memory GPU configuration
"""

import os
import sys
from pathlib import Path

# Add PipeServe to Python path
sys.path.append(str(Path(__file__).parent.parent.parent / "PipeServe"))

from BatchConfigurator import BatchConfigurator


def main():
    """Main function to run Llama2-13B optimization example"""
    
    print("=" * 60)
    print("Llama2-13B Batch Configuration Optimization Example")
    print("=" * 60)
    
    # Configuration file path
    script_dir = os.path.dirname(__file__)  # 获取当前脚本的目录
    config_file = os.path.join(script_dir, 'config.json')
    result_file = os.path.join(script_dir, 'result.json')

    print(f"Loading configuration from: {config_file}")
    
    try:
        # Create BatchConfigurator from JSON configuration
        configurator = BatchConfigurator.from_json_config(config_file)
        
        print("\nConfiguration loaded successfully!")
        print(f"Model: {configurator.analysis.model_config.name}")
        print(f"GPU: {configurator.analysis.gpu_config.name}")
        print(f"GPU Memory: {configurator.gpu.mem_per_GPU_in_GB}GB")
        print(f"SLO Prefill: {configurator.slo_p}s")
        print(f"SLO Decode: {configurator.slo_d}s")
        print(f"Max Chunk Size: {configurator.max_chunk_size}")
        
        # Display model information
        model_config = configurator.analysis.model_config
        print("\nModel Details:")
        print(f"  Layers: {model_config.num_layers}")
        print(f"  Hidden Dim: {model_config.hidden_dim}")
        print(f"  Attention Heads: {model_config.n_head}")
        print(f"  Vocabulary Size: {model_config.vocab_size}")
        
        print("\nStarting optimization...")
        print("Note: 13B model optimization may take longer due to increased complexity...")
        
        # Run optimization and save results to JSON (uses algorithm from config file)
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
            
            # Additional analysis for 13B model
            print("\nPerformance Analysis:")
            print(f"  DFS Count: {result.dfs_count}")
            print(f"  Original DFS Count: {result.dfs_original_count}")
            if result.dfs_original_count > 0:
                reduction = (1 - result.dfs_count / result.dfs_original_count) * 100
                print(f"  Search Space Reduction: {reduction:.1f}%")
            
        else:
            print("\nOptimization failed - no feasible solution found!")
            print("This could be due to:")
            print("1. Tight SLO constraints for the 13B model size")
            print("2. GPU memory limitations")
            print("3. Model configuration issues")
            print("\nSuggestions:")
            print("- Try relaxing SLO constraints (increase slo_prefill and slo_decode)")
            print("- Use a GPU with more memory")
            print("- Consider tensor parallelism (tp_size > 1)")
            
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found!")
        print("Make sure you're running this script from the example directory.")
    except Exception as e:
        print(f"Error during optimization: {str(e)}")
        print("\nDebugging tips:")
        print("- Check if all model config files exist")
        print("- Verify GPU configuration is valid")
        print("- Try running with verbose=true in config.json")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
