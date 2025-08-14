import argparse
import logging
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

# Add llm-analysis to Python path
sys.path.append(str(Path(__file__).parent / "llm-analysis"))

from llm_analysis.analysis import LLMAnalysis
from llm_analysis.config import (
    ParallelismConfig,
    get_dtype_config_by_name,
    get_gpu_config_by_name,
    get_model_config_by_name,
)
from llm_analysis.logger import logger

try:
    from .gpu_config import GPU, load_gpus_from_config
    from .ModelPartitioner import ModelPartitioner
    from .config_manager import BatchConfigParams, BatchOptimizationResult, ConfigManager
except ImportError:
    from gpu_config import GPU, load_gpus_from_config
    from ModelPartitioner import ModelPartitioner
    from config_manager import BatchOptimizationResult, ConfigManager


logger.setLevel(logging.ERROR)

MODEL_CONFIG_FILE: str = str(Path(__file__).parent / "model_configs")


class BatchConfigurator:
    """Batch configurator for optimizing LLM inference pipeline with SLO constraints

    This class handles batch size optimization and model partitioning to meet
    Service Level Objectives (SLO) for prefill and decode latencies.
    """

    def __init__(
        self,
        analysis: LLMAnalysis,
        slo_p: float = 2,
        slo_d: float = 0.5,
        max_chunk_size: int = 256,
    ) -> None:
        """Initialize the batch configurator

        Args:
            analysis (LLMAnalysis): LLM analysis object containing model and GPU configurations
            slo_p (float, optional): Prefill latency SLO in seconds. Defaults to 2.
            slo_d (float, optional): Decode latency SLO in seconds. Defaults to 0.5.
            max_chunk_size (int, optional): Maximum sequence chunk size. Defaults to 256.
        """
        self.analysis: LLMAnalysis = analysis
        self.slo_p: float = slo_p
        self.slo_d: float = slo_d
        self.max_chunk_size: int = max_chunk_size
        self.layers: int = analysis.model_config.num_layers
        self.gpu: GPU = load_gpus_from_config(analysis.gpu_config.name)

        self.delta: float = 100000
        self.G: List[int] = []

        self.max_stage: int = 0
        self.max_layer_per_gpu: int = 0

        self.dfs_count: int = 0
        self.dfs_original_count: int = 0

        # Initialize model partitioner
        self.partitioner: Optional[ModelPartitioner] = None

    def find_partition(self) -> Tuple[List[int], float]:
        """Find optimal model layer partitioning across pipeline stages

        Creates a ModelPartitioner instance with current parameters and finds
        the best way to distribute model layers across pipeline stages to
        minimize latency differences between prefill and decode operations.

        Returns:
            tuple: (G, delta) where G is the partition result list and delta is the minimum difference
        """
        # Create model partitioner with current parameters
        self.partitioner = ModelPartitioner(
            self.analysis, self.max_stage, self.max_layer_per_gpu
        )
        self.partitioner.single_layer_prefill_latency = (
            self.single_layer_prefill_latency
        )
        self.partitioner.single_layer_decode_latency = self.single_layer_decode_latency
        self.partitioner.transfer_p = self.transfer_p
        self.partitioner.transfer_d = self.transfer_d

        # Use the find_partition method from ModelPartitioner
        G, delta = self.partitioner.find_partition()

        # Update local state
        self.G = G
        self.delta = delta

        return self.G, self.delta

    def solve_brute_force(self) -> Optional[List[int]]:
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
        best_G: List[int] = []
        best_delta: float = 100000
        best_batch_d: int = 1
        best_batch_p: int = 1
        best_cost: float = 1000000

        # Iterate through prefill batch sizes
        for batch_p in range(1, 100):
            # Calculate GPU memory constraints for current batch size
            self.max_layer_per_gpu = self.max_layer_per_GPU(
                batch_p, self.max_chunk_size
            )

            # Determine number of pipeline stages needed
            self.max_stage = (
                self.layers // self.max_layer_per_gpu
                if self.layers % self.max_layer_per_gpu == 0
                else self.layers // self.max_layer_per_gpu + 1
            )
            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                / self.layers
            )

            # Early termination if prefill SLO is exceeded
            if (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                + (self.max_stage - 1) * self.transfer_p
                > self.slo_p
            ):
                break

            # Iterate through decode batch sizes
            for batch_d in range(1, 1000):
                self.transfer_d = self.transfer_time(batch_d, 1)
                # Early termination if decode SLO is exceeded
                if (
                    self.analysis.inference(batch_d, 1)["decode_latency"]
                    + (self.max_stage - 1) * self.transfer_d
                    > self.slo_d
                ):
                    break

                self.single_layer_decode_latency = (
                    self.analysis.inference(batch_d, 1)["decode_latency"] / self.layers
                )

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
        print(
            f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}"
        )
        if self.partitioner:
            print(
                f"dfs count: {self.partitioner.dfs_count}, original dfs count: {self.partitioner.dfs_original_count}"
            )
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)[
            "prefill_latency"
        ]
        decode_latency = self.analysis.inference(best_batch_d, 1)["decode_latency"]
        print(
            f"prefill latency: {prefill_latency + (self.max_stage - 1) * (self.transfer_p + best_delta)}, decode latency: {decode_latency + (self.max_stage - 1) * (self.transfer_d + best_delta)}"
        )

    def solve_bucket(self) -> Optional[List[int]]:
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
        best_G: List[int] = []
        best_delta: float = 100000
        best_batch_d: int = 1
        best_batch_p: int = 1
        best_cost: float = 1000000

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
            self.max_layer_per_gpu = self.max_layer_per_GPU(
                batch_p, self.max_chunk_size
            )

            # Determine pipeline stages needed
            self.max_stage = (
                self.layers // self.max_layer_per_gpu
                if self.layers % self.max_layer_per_gpu == 0
                else self.layers // self.max_layer_per_gpu + 1
            )
            if self.max_stage == 1:
                print(f"stages: 1, G: [{self.layers}]")
                return [self.layers]

            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                / self.layers
            )

            # Search through buckets in reverse order (prioritize higher batch sizes)
            for bucket in buckets[::-1]:
                batch_d = bucket[0]
                self.transfer_d = self.transfer_time(batch_d, 1)
                self.single_layer_decode_latency = (
                    self.analysis.inference(batch_d, 1)["decode_latency"] / self.layers
                )

                # Quick check if bucket is feasible
                G, delta = self.find_partition()
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue

                # Search within the feasible bucket
                for batch_d in range(bucket[0], bucket[1] + 1):
                    self.single_layer_decode_latency = (
                        self.analysis.inference(batch_d, 1)["decode_latency"]
                        / self.layers
                    )
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
        print(
            f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}"
        )
        if self.partitioner:
            print(
                f"dfs count: {self.partitioner.dfs_count}, original dfs count: {self.partitioner.dfs_original_count}"
            )
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)[
            "prefill_latency"
        ]
        decode_latency = self.analysis.inference(best_batch_d, 1)["decode_latency"]
        print(f"prefill latency: {prefill_latency}, decode latency: {decode_latency}")
        for i, layer in enumerate(best_G):
            print(
                f"Stage {i}: {layer} layers, prefill latency: {prefill_latency * best_G[i] / self.layers:.3f}, decode latency: {decode_latency * best_G[i] / self.layers:.3f}"
            )

    def max_batch(self) -> Tuple[int, int]:
        """Find maximum feasible batch sizes using binary search

        Uses binary search to efficiently find the maximum batch sizes for both
        prefill (batch_p) and decode (batch_d) operations that still meet the
        SLO constraints. This helps reduce the search space for optimization.

        Returns:
            tuple: (max_batch_p, max_batch_d) - maximum feasible batch sizes
        """
        # Binary search for maximum prefill batch size
        left_p: int = 1
        right_p: int = 99
        best_batch_p: int = 1

        while left_p <= right_p:
            mid_p = (left_p + right_p) // 2
            # Calculate GPU constraints and pipeline configuration
            self.max_layer_per_gpu = self.max_layer_per_GPU(mid_p, self.max_chunk_size)
            self.max_stage = (
                self.layers // self.max_layer_per_gpu
                if self.layers % self.max_layer_per_gpu == 0
                else self.layers // self.max_layer_per_gpu + 1
            )
            self.transfer_p = self.transfer_time(mid_p, self.max_chunk_size)

            # Calculate total prefill latency including transfer overhead
            prefill = (
                self.analysis.inference(mid_p, self.max_chunk_size)["prefill_latency"]
                + (self.max_stage - 1) * self.transfer_p
            )

            # Check if configuration meets prefill SLO
            if prefill <= self.slo_p:
                best_batch_p = mid_p
                left_p = mid_p + 1  # Try larger batch size
            else:
                right_p = mid_p - 1  # Try smaller batch size
        max_batch_p = best_batch_p

        # Binary search for maximum decode batch size
        left_d: int = 1
        right_d: int = 999
        best_batch_d: int = 1

        while left_d <= right_d:
            mid_d = (left_d + right_d) // 2
            # Calculate transfer overhead for decode phase
            self.transfer_d = self.transfer_time(mid_d, 1)

            # Calculate total decode latency including transfer overhead
            decode = (
                self.analysis.inference(mid_d, 1)["decode_latency"]
                + (self.max_stage - 1) * self.transfer_d
            )

            # Check if configuration meets decode SLO
            if decode <= self.slo_d:
                best_batch_d = mid_d
                left_d = mid_d + 1  # Try larger batch size
            else:
                right_d = mid_d - 1  # Try smaller batch size
        max_batch_d = best_batch_d

        print(
            f"Binary search found maximum batch sizes - Prefill: {max_batch_p}, Decode: {max_batch_d}"
        )
        return max_batch_p, max_batch_d

    def check_slo(self, batch_p: int, batch_d: int, delta: float) -> bool:
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
        if (
            self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
            + (self.max_stage - 1) * (self.transfer_p + delta)
            > self.slo_p
        ):
            return False
        if (
            self.analysis.inference(batch_d, 1)["decode_latency"]
            + (self.max_stage - 1) * (self.transfer_d + delta)
            > self.slo_d
        ):
            return False
        return True

    def cost(self, batch_p: int, batch_d: int, outnum: int = 129) -> float:
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
        return (
            self.single_layer_prefill_latency / batch_p
            + outnum * self.single_layer_decode_latency / batch_d
        ) * self.layers

    def max_layer_per_GPU(
        self, batch_size: int, seq_len: int, memory_buffer_gb: float = 5.0
    ) -> int:
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
        # Calculate memory requirements per layer
        weight_memory_per_layer = self.analysis.get_weight_memory_per_layer()
        memory_kv_cache_per_layer = self.analysis.get_memory_kv_cache_per_layer(
            batch_size=batch_size, seq_len=seq_len, kv_cache_dtype_bytes=2
        )
        activation_memory_per_layer = self.analysis.get_activation_memory_per_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            layernorm_dtype_bytes=2,
            is_inference=True,
        )

        # Total memory required per layer
        weight_per_layer = (
            weight_memory_per_layer
            + memory_kv_cache_per_layer
            + activation_memory_per_layer
        )

        # Calculate maximum layers that fit within GPU memory constraints
        max_layer = math.floor(
            (self.gpu.mem_per_GPU_in_GB - memory_buffer_gb)
            / (weight_per_layer / 1024 / 1024 / 1024)
        )
        return max_layer

    def transfer_time(self, batch_size: int, seq_len: int = 1) -> float:
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

    def test(self, batch_p: int = 1, batch_d: int = 1) -> None:
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
            prefill = self.analysis.inference(i, self.max_chunk_size)["prefill_latency"]
            prefill_latencys.append(prefill)

        # Collect decode latencies with step size of 20 for efficiency
        for i in range(1, batch_d + 1, 20):
            decode = self.analysis.inference(i, 1)["decode_latency"]
            decode_latencys.append(decode)

        print(
            f"prefill latency: {prefill_latencys},\n decode latency: {decode_latencys}"
        )

    @classmethod
    def from_json_config(cls, config_file_path: str) -> 'BatchConfigurator':
        """Create BatchConfigurator instance from JSON configuration file

        Args:
            config_file_path (str): Path to the JSON configuration file

        Returns:
            BatchConfigurator: Configured instance ready for optimization

        Example:
            >>> configurator = BatchConfigurator.from_json_config('./config.json')
            >>> result = configurator.optimize_to_json('./result.json')
        """
        # Load configuration from JSON
        config = ConfigManager.load_config_from_json(config_file_path)
        
        # Create analysis instance from config
        analysis = ConfigManager.create_analysis_from_config(config)
        
        # Create and return BatchConfigurator instance
        configurator = cls(
            analysis=analysis,
            slo_p=config.slo_prefill,
            slo_d=config.slo_decode,
            max_chunk_size=config.max_chunk_size
        )
        
        # Store config for later use
        configurator._config_params = config
        
        return configurator

    def optimize_to_json(
        self, 
        result_file_path: str, 
        algorithm: str = None
    ) -> BatchOptimizationResult:
        """Run batch optimization and save results to JSON file

        Args:
            result_file_path (str): Path to save the optimization results
            algorithm (str): Algorithm to use ("brute-force", "bucket", "both"). 
                           If None, uses the algorithm from config file

        Returns:
            BatchOptimizationResult: Optimization results

        Example:
            >>> with open('./config.json', 'r') as f:
            ...     configurator = BatchConfigurator.from_json_config('./config.json')
            >>> result = configurator.optimize_to_json('./result.json')
        """
        import time

        # Use algorithm from config if not specified
        if algorithm is None and hasattr(self, '_config_params'):
            algorithm = self._config_params.algorithm
        elif algorithm is None:
            algorithm = "both"  # fallback default

        # Initialize result tracking
        best_result = None
        total_start_time = time.time()
        
        print(f"Using optimization algorithm: {algorithm}")
        
        # Run algorithms based on selection
        if algorithm in ["brute-force", "both"]:
            print("\n=== Running Brute Force Algorithm ===")
            start_time = time.time()
            brute_force_result = self._run_brute_force_with_result()
            brute_force_time = time.time() - start_time
            
            if brute_force_result:
                brute_force_result.solve_time = brute_force_time
                best_result = brute_force_result
                print(f"Brute force solve time: {brute_force_time:.2f} s")

        if algorithm in ["bucket", "both"]:
            print("\n=== Running Bucket Optimization Algorithm ===")
            start_time = time.time()
            bucket_result = self._run_bucket_with_result()
            bucket_time = time.time() - start_time
            
            if bucket_result:
                bucket_result.solve_time = bucket_time
                # Choose the better result (lower cost)
                if best_result is None or bucket_result.best_cost < best_result.best_cost:
                    best_result = bucket_result
                print(f"Bucket solve time: {bucket_time:.2f} s")

        # Ensure we have a result
        if best_result is None:
            print("No valid configuration found!")
            return None

        # Add configuration parameters to result
        if hasattr(self, '_config_params'):
            from dataclasses import asdict
            best_result.config_params = asdict(self._config_params)
        
        # Save result to JSON file
        ConfigManager.save_result_to_json(best_result, result_file_path)
        
        print(f"\nOptimization completed! Results saved to: {result_file_path}")
        print(f"Total optimization time: {time.time() - total_start_time:.2f} s")
        
        return best_result

    def _run_brute_force_with_result(self) -> Optional[BatchOptimizationResult]:
        """Run brute force optimization and return structured result"""
        # Initialize best configuration tracking
        best_G: List[int] = []
        best_delta: float = 100000
        best_batch_d: int = 1
        best_batch_p: int = 1
        best_cost: float = 1000000
        
        found_solution = False

        # Iterate through prefill batch sizes
        for batch_p in range(1, 100):
            # Calculate GPU memory constraints for current batch size
            self.max_layer_per_gpu = self.max_layer_per_GPU(
                batch_p, self.max_chunk_size
            )

            # Determine number of pipeline stages needed
            self.max_stage = (
                self.layers // self.max_layer_per_gpu
                if self.layers % self.max_layer_per_gpu == 0
                else self.layers // self.max_layer_per_gpu + 1
            )
            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                / self.layers
            )

            # Early termination if prefill SLO is exceeded
            if (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                + (self.max_stage - 1) * self.transfer_p
                > self.slo_p
            ):
                break

            # Iterate through decode batch sizes
            for batch_d in range(1, 1000):
                self.transfer_d = self.transfer_time(batch_d, 1)
                # Early termination if decode SLO is exceeded
                if (
                    self.analysis.inference(batch_d, 1)["decode_latency"]
                    + (self.max_stage - 1) * self.transfer_d
                    > self.slo_d
                ):
                    break

                self.single_layer_decode_latency = (
                    self.analysis.inference(batch_d, 1)["decode_latency"] / self.layers
                )

                # Handle single stage case
                if self.max_stage == 1:
                    print(f"stages: 1, G: [{self.layers}]")
                    found_solution = True
                    best_G = [self.layers]
                    best_batch_p = batch_p
                    best_batch_d = batch_d
                    best_delta = 0
                    best_cost = self.cost(batch_p, batch_d)
                    break

                # Find optimal partition for current batch configuration
                G, delta = self.find_partition()
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue

                found_solution = True
                # Update best configuration if cost is improved
                cost = self.cost(batch_p, batch_d)
                if cost < best_cost:
                    best_batch_d = batch_d
                    best_batch_p = batch_p
                    best_G = deepcopy(G)
                    best_delta = delta
                    best_cost = cost

            if self.max_stage == 1 and found_solution:
                break
        
        if not found_solution:
            return None
            
        # Calculate latencies
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)["prefill_latency"]
        decode_latency = self.analysis.inference(best_batch_d, 1)["decode_latency"]
        total_prefill_latency = prefill_latency + (self.max_stage - 1) * (self.transfer_p + best_delta)
        total_decode_latency = decode_latency + (self.max_stage - 1) * (self.transfer_d + best_delta)
        
        # Create result object
        result = BatchOptimizationResult(
            best_batch_p=best_batch_p,
            best_batch_d=best_batch_d,
            best_partition=best_G,
            best_delta=best_delta,
            best_cost=best_cost,
            max_stage=self.max_stage,
            max_layer_per_gpu=self.max_layer_per_gpu,
            prefill_latency=prefill_latency,
            decode_latency=decode_latency,
            total_prefill_latency=total_prefill_latency,
            total_decode_latency=total_decode_latency,
            dfs_count=self.partitioner.dfs_count if self.partitioner else 0,
            dfs_original_count=self.partitioner.dfs_original_count if self.partitioner else 0,
            meets_slo=self.check_slo(best_batch_p, best_batch_d, best_delta)
        )
        
        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        
        return result

    def _run_bucket_with_result(self) -> Optional[BatchOptimizationResult]:
        """Run bucket optimization and return structured result"""
        # Initialize best configuration tracking
        best_G: List[int] = []
        best_delta: float = 100000
        best_batch_d: int = 1
        best_batch_p: int = 1
        best_cost: float = 1000000
        
        found_solution = False

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
            self.max_layer_per_gpu = self.max_layer_per_GPU(
                batch_p, self.max_chunk_size
            )

            # Determine pipeline stages needed
            self.max_stage = (
                self.layers // self.max_layer_per_gpu
                if self.layers % self.max_layer_per_gpu == 0
                else self.layers // self.max_layer_per_gpu + 1
            )
            if self.max_stage == 1:
                print(f"stages: 1, G: [{self.layers}]")
                found_solution = True
                best_G = [self.layers]
                best_batch_p = batch_p
                best_batch_d = 1
                best_delta = 0
                best_cost = self.cost(batch_p, 1)
                break

            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = (
                self.analysis.inference(batch_p, self.max_chunk_size)["prefill_latency"]
                / self.layers
            )

            # Search through buckets in reverse order (prioritize higher batch sizes)
            for bucket in buckets[::-1]:
                batch_d = bucket[0]
                self.transfer_d = self.transfer_time(batch_d, 1)
                self.single_layer_decode_latency = (
                    self.analysis.inference(batch_d, 1)["decode_latency"] / self.layers
                )

                # Quick check if bucket is feasible
                G, delta = self.find_partition()
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue

                # Search within the feasible bucket
                for batch_d in range(bucket[0], bucket[1] + 1):
                    self.single_layer_decode_latency = (
                        self.analysis.inference(batch_d, 1)["decode_latency"]
                        / self.layers
                    )
                    G, delta = self.find_partition()

                    if not self.check_slo(batch_p, batch_d, delta):
                        continue

                    found_solution = True
                    # Update best configuration if cost is improved
                    cost = self.cost(batch_p, batch_d)
                    if cost < best_cost:
                        best_batch_d = batch_d
                        best_batch_p = batch_p
                        best_G = deepcopy(G)
                        best_delta = delta
                        best_cost = cost
                break
        
        if not found_solution:
            return None
            
        # Calculate latencies
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)["prefill_latency"]
        decode_latency = self.analysis.inference(best_batch_d, 1)["decode_latency"]
        total_prefill_latency = prefill_latency + (self.max_stage - 1) * (self.transfer_p + best_delta)
        total_decode_latency = decode_latency + (self.max_stage - 1) * (self.transfer_d + best_delta)
        
        # Create result object
        result = BatchOptimizationResult(
            best_batch_p=best_batch_p,
            best_batch_d=best_batch_d,
            best_partition=best_G,
            best_delta=best_delta,
            best_cost=best_cost,
            max_stage=self.max_stage,
            max_layer_per_gpu=self.max_layer_per_gpu,
            prefill_latency=prefill_latency,
            decode_latency=decode_latency,
            total_prefill_latency=total_prefill_latency,
            total_decode_latency=total_decode_latency,
            dfs_count=self.partitioner.dfs_count if self.partitioner else 0,
            dfs_original_count=self.partitioner.dfs_original_count if self.partitioner else 0,
            meets_slo=self.check_slo(best_batch_p, best_batch_d, best_delta)
        )
        
        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        
        return result

    @staticmethod
    def create_sample_config(output_path: str = "config_sample.json") -> None:
        """Create a sample configuration JSON file

        Args:
            output_path (str): Path to save the sample configuration file

        Example:
            >>> BatchConfigurator.create_sample_config('./my_config.json')
        """
        ConfigManager.create_sample_config(output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for batch configuration

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Batch Configurator for LLM Inference Pipeline Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration arguments
    parser.add_argument(
        "--model", type=str, default="Llama-2-7b", help="Model name to use for analysis"
    )

    # Hardware configuration arguments
    parser.add_argument(
        "--gpu-name",
        type=str,
        default="t4-pcie-16gb",
        choices=["t4-pcie-16gb", "rtx6000-48gb", "a100-40gb", "a100-80gb"],
        help="GPU model name",
    )

    # Data type configuration arguments
    parser.add_argument(
        "--dtype",
        type=str,
        default="w16a16e16",
        choices=[
            "w16a16e16",
            "w16a16e32",
            "w4a16e16",
            "w4a16e32",
            "w4a4e16",
            "w4a4e32",
            "w4a8e16",
            "w8a8e16",
        ],
        help="Data type configuration (weight-activation-embedding bits)",
    )

    # Sequence and batch configuration arguments
    parser.add_argument(
        "--seq-len", type=int, default=1024, help="Sequence length for analysis"
    )

    parser.add_argument(
        "--batch-size", type=int, default=2, help="Base batch size for analysis"
    )

    parser.add_argument(
        "--max-chunk-size", type=int, default=256, help="Maximum sequence chunk size"
    )

    # SLO constraint arguments
    parser.add_argument(
        "--slo-prefill",
        type=float,
        default=1.5,
        help="Prefill latency SLO constraint in seconds",
    )

    parser.add_argument(
        "--slo-decode",
        type=float,
        default=0.5,
        help="Decode latency SLO constraint in seconds",
    )

    # Parallelism configuration arguments
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )

    parser.add_argument(
        "--pp-size", type=int, default=1, help="Pipeline parallelism size"
    )

    parser.add_argument(
        "--sp-size", type=int, default=1, help="Sequence parallelism size"
    )

    parser.add_argument("--dp-size", type=int, default=1, help="Data parallelism size")

    # Algorithm selection arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        default="both",
        choices=["brute-force", "bucket", "both"],
        help="Algorithm to use for batch configuration optimization",
    )

    # Test mode arguments
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode to analyze latency patterns",
    )

    parser.add_argument(
        "--test-batch-p",
        type=int,
        default=8,
        help="Maximum prefill batch size for testing",
    )

    parser.add_argument(
        "--test-batch-d",
        type=int,
        default=500,
        help="Maximum decode batch size for testing",
    )

    # Output control arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--quiet", action="store_true", help="Suppress non-essential output"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Configure logging based on verbosity
    if args.quiet:
        logger.setLevel(logging.CRITICAL)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    # Configuration parameters for LLM analysis from command line
    model = args.model
    print(f"Model: {model}")
    model_name = str(Path(MODEL_CONFIG_FILE) / model)
    gpu_name = args.gpu_name
    dtype_name = args.dtype
    seq_len = args.seq_len
    batch_size = args.batch_size

    print("Configuration:")
    print(f"  GPU: {gpu_name}")
    print(f"  Data type: {dtype_name}")
    print(f"  Base batch size: {batch_size}")
    print(f"  Max chunk size: {args.max_chunk_size}")
    print(f"  SLO - Prefill: {args.slo_prefill}s, Decode: {args.slo_decode}s")
    print(
        f"  Parallelism - TP: {args.tp_size}, PP: {args.pp_size}, SP: {args.sp_size}, DP: {args.dp_size}"
    )

    # Load hardware configuration
    gpu = load_gpus_from_config(gpu_name)

    # Initialize model, GPU, and data type configurations
    model_config = get_model_config_by_name(model_name)
    gpu_config = get_gpu_config_by_name(gpu_name)
    dtype_config = get_dtype_config_by_name(dtype_name)
    parallel_config = ParallelismConfig(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        sp_size=args.sp_size,
        dp_size=args.dp_size,
    )

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
    batch_configurator = BatchConfigurator(
        analysis,
        slo_p=args.slo_prefill,
        slo_d=args.slo_decode,
        max_chunk_size=args.max_chunk_size,
    )

    # Run in test mode if specified
    if args.test_mode:
        print("\n=== Running Test Mode ===")
        print(
            f"Testing batch sizes - Prefill: 1-{args.test_batch_p}, Decode: 1-{args.test_batch_d} (step 20)"
        )
        batch_configurator.test(batch_p=args.test_batch_p, batch_d=args.test_batch_d)
        exit(0)

    # Run optimization algorithms based on selection
    if args.algorithm in ["brute-force", "both"]:
        print("\n=== Running Brute Force Algorithm ===")
        start_time = time.time()
        batch_configurator.solve_brute_force()
        brute_force_time = time.time() - start_time
        print(f"Brute force solve time: {brute_force_time:.2f} s")

    if args.algorithm in ["bucket", "both"]:
        print("\n=== Running Bucket Optimization Algorithm ===")
        start_time = time.time()
        batch_configurator.solve_bucket()
        bucket_time = time.time() - start_time
        print(f"Bucket solve time: {bucket_time:.2f} s")

    # Performance comparison if both algorithms were run
    if args.algorithm == "both":
        print("\n=== Performance Comparison ===")
        speedup = brute_force_time / bucket_time if bucket_time > 0 else float("inf")
        print(f"Bucket optimization speedup: {speedup:.2f}x")
