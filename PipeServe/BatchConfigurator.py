
from llm_analysis.analysis import infer, LLMAnalysis
import os
import math
import time
import json
from pathlib import Path
from transformers import AutoConfig
from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 ParallelismConfig, get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name)
from copy import deepcopy
import logging
from llm_analysis.logger import logger
try:
    from .gpu_config import load_gpus_from_config
    from .ModelPartitioner import ModelPartitioner
except ImportError:
    from gpu_config import load_gpus_from_config
    from ModelPartitioner import ModelPartitioner


logger.setLevel(logging.ERROR)

MODEL_CONFIG_FILE = str(Path(__file__).parent / "model_configs")


class BatchConfigurator():
    """Batch configurator for optimizing LLM inference pipeline with SLO constraints
    
    This class handles batch size optimization and model partitioning to meet
    Service Level Objectives (SLO) for prefill and decode latencies.
    """

    def __init__(self, analysis, slo_p=2, slo_d=0.5, max_chunk_size=256):
        """Initialize the batch configurator
        
        Args:
            analysis (LLMAnalysis): LLM analysis object containing model and GPU configurations
            slo_p (float, optional): Prefill latency SLO in seconds. Defaults to 2.
            slo_d (float, optional): Decode latency SLO in seconds. Defaults to 0.5.
            max_chunk_size (int, optional): Maximum sequence chunk size. Defaults to 256.
        """
        self.analysis = analysis
        self.slo_p = slo_p
        self.slo_d = slo_d
        self.max_chunk_size = max_chunk_size
        self.layers = analysis.model_config.num_layers
        self.gpu = load_gpus_from_config(analysis.gpu_config.name)
        
        self.delta = 100000
        self.G = []
        
        self.max_stage = 0
        self.max_layer_per_gpu = 0
        
        self.dfs_count = 0
        self.dfs_original_count = 0
        
        # Initialize model partitioner
        self.partitioner = None

    def find_partition(self):
        """Find optimal model layer partitioning across pipeline stages
        
        Creates a ModelPartitioner instance with current parameters and finds
        the best way to distribute model layers across pipeline stages to
        minimize latency differences between prefill and decode operations.
        
        Returns:
            tuple: (G, delta) where G is the partition result list and delta is the minimum difference
        """
        # Create model partitioner with current parameters
        self.partitioner = ModelPartitioner(self.analysis, self.max_stage, self.max_layer_per_gpu)
        self.partitioner.single_layer_prefill_latency = self.single_layer_prefill_latency
        self.partitioner.single_layer_decode_latency = self.single_layer_decode_latency
        self.partitioner.transfer_p = self.transfer_p
        self.partitioner.transfer_d = self.transfer_d
        
        # Use the find_partition method from ModelPartitioner
        G, delta = self.partitioner.find_partition()
        
        # Update local state
        self.G = G
        self.delta = delta
        
        return self.G, self.delta

    def solve_brute_force(self):
        """Solve batch configuration using brute force search approach
        
        Iterates through all combinations of batch_p (prefill batch size) and 
        batch_d (decode batch size) to find the optimal configuration that
        meets SLO constraints while minimizing cost.
        
        The method explores batch sizes from 1 to 100 for prefill and 1 to 1000
        for decode, breaking early when SLO constraints are exceeded.
        
        Returns:
            list: Optimal partition configuration if found for single stage, None otherwise
        """
        # Initialize best configuration tracking
        best_G = []
        best_delta = 100000
        best_batch_d = 1
        best_batch_p = 1
        best_cost = 1000000
        
        # Iterate through prefill batch sizes
        for batch_p in range(1, 100):
            # Calculate GPU memory constraints for current batch size
            self.max_layer_per_gpu = self.max_layer_per_GPU(batch_p, self.max_chunk_size)

            # Determine number of pipeline stages needed
            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] / self.layers

            # Early termination if prefill SLO is exceeded
            if self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * self.transfer_p > self.slo_p:
                break

            # Iterate through decode batch sizes
            for batch_d in range(1, 1000):
                self.transfer_d = self.transfer_time(batch_d, 1)
                # Early termination if decode SLO is exceeded
                if self.analysis.inference(batch_d, 1)['decode_latency'] + (self.max_stage - 1) * self.transfer_d > self.slo_d:
                    break
                
                self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers

                # Handle single stage case
                if self.max_stage == 1:
                    print(f"stages: 1, G: [{self.layers}]")
                    return [self.layers]

                # Find optimal partition for current batch configuration
                G, delta = self.find_partition()
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue
                    
                # Update best configuration if cost is improved
                cost = self.cost(batch_p, batch_d)
                if cost < best_cost:
                    best_batch_d = batch_d
                    best_batch_p = batch_p
                    best_G = deepcopy(G)
                    best_delta = delta
                    best_cost = cost

        # Output results
        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        if self.partitioner:
            print(f"dfs count: {self.partitioner.dfs_count}, original dfs count: {self.partitioner.dfs_original_count}")
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)['prefill_latency']
        decode_latency = self.analysis.inference(best_batch_d, 1)['decode_latency']
        print(f"prefill latency: {prefill_latency + (self.max_stage - 1) * (self.transfer_p + best_delta)}, decode latency: {decode_latency + (self.max_stage - 1) * (self.transfer_d + best_delta)}")
        
    def solve_bucket(self):
        """Solve batch configuration using optimized bucket search approach
        
        Uses binary search to find maximum feasible batch sizes, then employs
        a bucketing strategy to reduce search space. Batch_d values are grouped
        into buckets of size 10, and the search is performed in reverse order
        to prioritize higher batch sizes.
        
        This method is more efficient than brute force as it reduces the number
        of configurations that need to be evaluated while still finding optimal solutions.
        
        Returns:
            list: Optimal partition configuration if found for single stage, None otherwise
        """
        # Initialize best configuration tracking
        best_G = []
        best_delta = 100000
        best_batch_d = 1
        best_batch_p = 1
        best_cost = 1000000
        
        # Get maximum feasible batch sizes using binary search
        max_batch_p, max_batch_d = self.max_batch()
        
        # Create buckets for batch_d values to reduce search space
        bucket_size = 10
        buckets = []
        for i in range(1, max_batch_d + 1, bucket_size):
            buckets.append((i, min(i + bucket_size - 1, max_batch_d)))
        
        # Linear search through prefill batch sizes
        for batch_p in range(1, max_batch_p + 1):
            # Calculate GPU memory constraints
            self.max_layer_per_gpu = self.max_layer_per_GPU(batch_p, self.max_chunk_size)

            # Determine pipeline stages needed
            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            if self.max_stage == 1:
                print(f"stages: 1, G: [{self.layers}]")
                return [self.layers]

            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] / self.layers

            # Search through buckets in reverse order (prioritize higher batch sizes)
            for bucket in buckets[::-1]:
                batch_d = bucket[0]
                self.transfer_d = self.transfer_time(batch_d, 1)
                self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers
                
                # Quick check if bucket is feasible
                G, delta = self.find_partition()
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue
                    
                # Search within the feasible bucket
                for batch_d in range(bucket[0], bucket[1] + 1):
                    self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers
                    G, delta = self.find_partition()
                    
                    if not self.check_slo(batch_p, batch_d, delta):
                        continue
                    
                    # Update best configuration if cost is improved
                    cost = self.cost(batch_p, batch_d)
                    if cost < best_cost:
                        best_batch_d = batch_d
                        best_batch_p = batch_p
                        best_G = deepcopy(G)
                        best_delta = delta
                        best_cost = cost
                break
                
        # Output results
        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        if self.partitioner:
            print(f"dfs count: {self.partitioner.dfs_count}, original dfs count: {self.partitioner.dfs_original_count}")
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)['prefill_latency']
        decode_latency = self.analysis.inference(best_batch_d, 1)['decode_latency']
        print(f"prefill latency: {prefill_latency}, decode latency: {decode_latency}")
        for i, layer in enumerate(best_G):
            print(f"Stage {i}: {layer} layers, prefill latency: {prefill_latency * best_G[i] / self.layers:.3f}, decode latency: {decode_latency * best_G[i] / self.layers:.3f}")

    def max_batch(self):
        """Find maximum feasible batch sizes using binary search
        
        Uses binary search to efficiently find the maximum batch sizes for both
        prefill (batch_p) and decode (batch_d) operations that still meet the
        SLO constraints. This helps reduce the search space for optimization.
        
        Returns:
            tuple: (max_batch_p, max_batch_d) - maximum feasible batch sizes
        """
        # Binary search for maximum prefill batch size
        left_p, right_p = 1, 99
        best_batch_p = 1
        
        while left_p <= right_p:
            mid_p = (left_p + right_p) // 2
            # Calculate GPU constraints and pipeline configuration
            self.max_layer_per_gpu = self.max_layer_per_GPU(mid_p, self.max_chunk_size)
            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            self.transfer_p = self.transfer_time(mid_p, self.max_chunk_size)
            
            # Calculate total prefill latency including transfer overhead
            prefill = self.analysis.inference(mid_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * self.transfer_p
            
            # Check if configuration meets prefill SLO
            if prefill <= self.slo_p:
                best_batch_p = mid_p
                left_p = mid_p + 1  # Try larger batch size
            else:
                right_p = mid_p - 1  # Try smaller batch size
        max_batch_p = best_batch_p

        # Binary search for maximum decode batch size
        left_d, right_d = 1, 999
        best_batch_d = 1
        
        while left_d <= right_d:
            mid_d = (left_d + right_d) // 2
            # Calculate transfer overhead for decode phase
            self.transfer_d = self.transfer_time(mid_d, 1)
            
            # Calculate total decode latency including transfer overhead
            decode = self.analysis.inference(mid_d, 1)['decode_latency'] + (self.max_stage - 1) * self.transfer_d
            
            # Check if configuration meets decode SLO
            if decode <= self.slo_d:
                best_batch_d = mid_d
                left_d = mid_d + 1  # Try larger batch size
            else:
                right_d = mid_d - 1  # Try smaller batch size
        max_batch_d = best_batch_d
        
        print(f"Binary search found maximum batch sizes - Prefill: {max_batch_p}, Decode: {max_batch_d}")
        return max_batch_p, max_batch_d

    def check_slo(self, batch_p, batch_d, delta):
        """Check if given batch configuration meets SLO constraints
        
        Validates whether the specified batch sizes and partition delta
        satisfy both prefill and decode latency SLO requirements.
        
        Args:
            batch_p (int): Prefill batch size
            batch_d (int): Decode batch size  
            delta (float): Latency difference from partitioning
            
        Returns:
            bool: True if SLO constraints are met, False otherwise
        """
        if self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * (self.transfer_p + delta) > self.slo_p:
            return False
        if self.analysis.inference(batch_d, 1)['decode_latency'] + (self.max_stage - 1) * (self.transfer_d + delta) > self.slo_d:
            return False
        return True
    
    def cost(self, batch_p, batch_d, outnum=129):
        """Calculate the total cost for given batch configuration
        
        Computes the cost based on prefill and decode latencies normalized
        by their respective batch sizes, considering the number of output tokens.
        
        Args:
            batch_p (int): Prefill batch size
            batch_d (int): Decode batch size
            outnum (int, optional): Number of output tokens. Defaults to 129.
            
        Returns:
            float: Total cost for the configuration
        """
        return (self.single_layer_prefill_latency / batch_p + outnum * self.single_layer_decode_latency / batch_d) * self.layers

    def max_layer_per_GPU(self,
                          batch_size,
                          seq_len,
                          memory_buffer_gb: float = 5.0):
        """Calculate maximum number of model layers that can fit on a single GPU
        
        Determines the maximum number of transformer layers that can be allocated
        to a single GPU based on memory constraints including model weights,
        KV cache, and activation memory requirements.
        
        Args:
            batch_size (int): Batch size for inference
            seq_len (int): Sequence length
            memory_buffer_gb (float, optional): Memory buffer in GB to reserve. Defaults to 5.0.
            
        Returns:
            int: Maximum number of layers that can fit on one GPU
        """
        import math
        
        # Calculate memory requirements per layer
        weight_memory_per_layer = self.analysis.get_weight_memory_per_layer()
        memory_kv_cache_per_layer = self.analysis.get_memory_kv_cache_per_layer(
            batch_size=batch_size, seq_len=seq_len, kv_cache_dtype_bytes=2)
        activation_memory_per_layer = self.analysis.get_activation_memory_per_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            layernorm_dtype_bytes=2,
            is_inference=True)

        # Total memory required per layer
        weight_per_layer = weight_memory_per_layer + memory_kv_cache_per_layer + activation_memory_per_layer

        # Calculate maximum layers that fit within GPU memory constraints
        max_layer = math.floor(
            (self.gpu.mem_per_GPU_in_GB - memory_buffer_gb) /
            (weight_per_layer / 1024 / 1024 / 1024))
        return max_layer

    def transfer_time(self, batch_size, seq_len=1):
        """Calculate data transfer time between pipeline stages
        
        Estimates the time required to transfer intermediate activations
        between different pipeline stages based on batch size and sequence length.
        Currently returns a fixed value but includes commented logic for
        more sophisticated calculations based on data size.
        
        Args:
            batch_size (int): Batch size for the operation
            seq_len (int, optional): Sequence length. Defaults to 1.
            
        Returns:
            float: Transfer time in seconds
        """
        # DECODE_LATENCY = 0.015
        # # DECODE_LATENCY = 0.005
        # SLOPE = 0.0274
        # INTERCEPT = -0.0304
        # # return DECODE_LATENCY
        # if seq_len == 1:  # decode
        #     return DECODE_LATENCY
        # else:
        #     return SLOPE * (self.analysis.model_config.hidden_dim *
        #                     batch_size * seq_len * 2 / 1024 / 1024) + INTERCEPT
        return 0.05

    def test(self, batch_p=1, batch_d=1):
        """Test method to analyze latency patterns across different batch sizes
        
        Generates latency measurements for various prefill and decode batch sizes
        to help understand performance characteristics and validate configurations.
        
        Args:
            batch_p (int, optional): Maximum prefill batch size to test. Defaults to 1.
            batch_d (int, optional): Maximum decode batch size to test. Defaults to 1.
            
        Prints:
            Prefill and decode latency arrays for the tested batch size ranges
        """
        prefill_latencys = []
        decode_latencys = []
        
        # Collect prefill latencies for batch sizes 1 to batch_p
        for i in range(1, batch_p + 1):
            prefill = self.analysis.inference(i, self.max_chunk_size)['prefill_latency']
            prefill_latencys.append(prefill)
            
        # Collect decode latencies with step size of 20 for efficiency
        for i in range(1, batch_d + 1, 20):
            decode = self.analysis.inference(i, 1)['decode_latency']
            decode_latencys.append(decode)
            
        print(f"prefill latency: {prefill_latencys},\n decode latency: {decode_latencys}")


if __name__ == "__main__":
    # Configuration parameters for LLM analysis
    model = "Llama-2-7b"
    print(f"model: {model}")
    model_name = str(Path(MODEL_CONFIG_FILE) / model)
    gpu_name = "t4-pcie-16gb"
    dtype_name = "w16a16e16"
    seq_len = 1024
    batch_size = 2

    # Load hardware configuration
    gpu = load_gpus_from_config(gpu_name)

    # Initialize model, GPU, and data type configurations
    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(tp_size=1,
                                        pp_size=1,
                                        sp_size=1,
                                        dp_size=1)

    # Create LLM analysis instance with all configurations
    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        flops_efficiency=gpu.flops_efficiency,
        hbm_memory_efficiency=gpu.hbm_memory_efficiency,
    )
    
    # Initialize batch configurator with SLO constraints
    BatchConfigurator = BatchConfigurator(analysis, slo_p=1.5, slo_d=0.5, max_chunk_size=256)
    
    # Solve using brute force approach and measure execution time
    start_time = time.time()
    BatchConfigurator.solve_brute_force()
    print(f"brute force solve time: {time.time() - start_time:.2f} s")
    
    # Solve using bucket optimization approach
    start_time = time.time()
    BatchConfigurator.solve_bucket()
    print(f"bucket solve time: {time.time() - start_time:.2f} s")
