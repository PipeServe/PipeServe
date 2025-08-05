from llm_analysis.analysis import infer, LLMAnalysis
import os
import math
import time
import json
from pathlib import Path
from transformers import AutoConfig
# from llm_analysis.analysis import LLMAnalysis
from llm_analysis.config import (DtypeConfig, GPUConfig, ModelConfig,
                                 ParallelismConfig, get_dtype_config_by_name,
                                 get_gpu_config_by_name,
                                 get_model_config_by_name)
from copy import deepcopy
import logging
from llm_analysis.logger import logger

logger.setLevel(logging.ERROR)

GPU_CONFIG_FILE = "gpu_config.json"
MODEL_CONFIG_FILE = f"{Path(__file__).parent}\\model_config\\"


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


class solver():

    def __init__(self, analysis, slo_p=2, slo_d=0.5, max_chunk_size=256):
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

    def dfs(self, G, stage, layers, max_diff, prev):
        """_summary_

        Args:
            G (list): 切分结果
            stage (_type_): 当前stage
            layers (_type_): 已经分配的layer数量
            max_diff (_type_): 之前累计stage的prefill和decode延迟差值最大值
            prev (_type_): 上一个stage的layer数量
        return:
            None
        """
        self.dfs_count += 1
        if stage == self.max_stage - 1:
            layer_now = self.layers - layers
            if layer_now > 0 and self.max_layer_per_gpu >= layer_now and layer_now <= prev:
                # update temp G and delta
                G_ = deepcopy(G)
                G_.append(layer_now)
                new_diff = max(
                    max_diff,
                    abs(layer_now * self.single_layer_prefill_latency + self.transfer_p - (self.single_layer_decode_latency * G[-1] + self.transfer_d)))
                # update global delta and G
                if new_diff < self.delta:
                    self.delta = new_diff
                    self.G = G_
            return

        upper = min(self.layers - layers, self.max_layer_per_gpu, prev)
        for layer_now in range(1, upper + 1):
            # update temp G and delta
            G_ = deepcopy(G)
            G_.append(layer_now)
            if stage > 0:
                new_diff = max(
                    max_diff,
                    abs(layer_now * self.single_layer_prefill_latency + self.transfer_p - (self.single_layer_decode_latency * G[-1] + self.transfer_d)))
            else:
                new_diff = max_diff
            self.dfs(G_, stage + 1, layers + layer_now, new_diff, layer_now)


    def dfs_original(self, G, stage, layers, max_diff, prev):
        """_summary_

        Args:
            G (list): 切分结果
            stage (_type_): 当前stage
            layers (_type_): 已经分配的layer数量
            max_diff (_type_): 之前累计stage的prefill和decode延迟差值最大值
        return:
            None
        """
        self.dfs_original_count += 1
        if stage == self.max_stage - 1:
            layer_now = self.layers - layers
            if layer_now > 0 and self.max_layer_per_gpu >= layer_now:
                # update temp G and delta
                G_ = deepcopy(G)
                G_.append(layer_now)
                new_diff = max(
                    max_diff,
                    abs(layer_now * self.single_layer_prefill_latency + self.transfer_p - (self.single_layer_decode_latency * G[-1] + self.transfer_d)))
                # update global delta and G
                if new_diff < self.delta:
                    self.delta = new_diff
                    self.G = G_
            return

        upper = min(self.layers - layers, self.max_layer_per_gpu)
        for layer_now in range(1, upper + 1):
            # update temp G and delta
            G_ = deepcopy(G)
            G_.append(layer_now)
            if stage > 0:
                new_diff = max(
                    max_diff,
                    abs(layer_now * self.single_layer_prefill_latency + self.transfer_p - (self.single_layer_decode_latency * G[-1] + self.transfer_d)))
            else:
                new_diff = max_diff
            self.dfs_original(G_, stage + 1, layers + layer_now, new_diff, layer_now)

    def find_partition(self):
        self.G = []
        self.delta = 100000
        
        upper = self.layers - (self.max_stage - 1)
        self.dfs_original(self.G, 0, 0, 0, upper)
        # self.dfs(self.G, 0, 0, 0, upper)
        
        return self.G, self.delta

    def solve_brute_force(self):
        """
        遍历batch_p和batch_d的组合，找到满足SLO的切分方案
        """
        best_G = []
        best_delta = 100000
        best_batch_d = 1
        best_batch_p = 1
        best_cost = 1000000
        for batch_p in range(1, 100):
            
            self.max_layer_per_gpu = self.max_layer_per_GPU(batch_p, self.max_chunk_size)

            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            self.single_layer_prefill_latency = self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] / self.layers

            if self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * self.transfer_p > self.slo_p:
                # print(f"batch_p: {batch_p} exceeds SLO, break")
                break

            for batch_d in range(1, 1000):
                self.transfer_d = self.transfer_time(batch_d, 1)
                if self.analysis.inference(batch_d, 1)['decode_latency'] + (self.max_stage - 1) * self.transfer_d > self.slo_d:
                    # print(f"batch_d: {batch_d} exceeds SLO, break")
                    break
                
                self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers

                if self.max_stage == 1:
                    print(f"stages: 1, G: [{self.layers}]")
                    return [self.layers]

                G, delta = self.find_partition()
                # print(f"batch_p: {batch_p}, batch_d: {batch_d}, G: {G}, delta: {delta}")
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue
                cost = self.cost(batch_p, batch_d)
                if cost < best_cost:
                    best_batch_d = batch_d
                    best_batch_p = batch_p
                    best_G = deepcopy(G)
                    best_delta = delta
                    best_cost = cost

        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        print(f"dfs count: {self.dfs_count}, original dfs count: {self.dfs_original_count}")
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)['prefill_latency']
        decode_latency = self.analysis.inference(best_batch_d, 1)['decode_latency']
        print(f"prefill latency: {prefill_latency + (self.max_stage - 1) * (self.transfer_p + best_delta)}, decode latency: {decode_latency + (self.max_stage - 1) * (self.transfer_d + best_delta)}")
        
    def solve_bucket(self):
        
        best_G = []
        best_delta = 100000
        best_batch_d = 1
        best_batch_p = 1
        best_cost = 1000000
        
        max_batch_p, max_batch_d = self.max_batch()
        # 将 batch_d 分桶，每个桶是连续的 batch_d 值
        bucket_size = 10
        buckets = []
        for i in range(1, max_batch_d + 1, bucket_size):
            buckets.append((i, min(i + bucket_size - 1, max_batch_d)))
        # print(f"batch_d 分桶结果: {buckets}")
        
        # 线性搜索 batch_p
        for batch_p in range(1, max_batch_p + 1):
            self.max_layer_per_gpu = self.max_layer_per_GPU(batch_p, self.max_chunk_size)

            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            if self.max_stage == 1:
                print(f"stages: 1, G: [{self.layers}]")
                return [self.layers]

            self.transfer_p = self.transfer_time(batch_p, self.max_chunk_size)
            
            self.single_layer_prefill_latency = self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] / self.layers

            for bucket in buckets[::-1]:
                batch_d = bucket[0]
                self.transfer_d = self.transfer_time(batch_d, 1)
                
                self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers
                
                G, delta = self.find_partition()
                # print(f"batch_p: {batch_p}, batch_d: {batch_d}, G: {G}, delta: {delta}")
                if len(G) == 0:
                    continue
                if not self.check_slo(batch_p, batch_d, delta):
                    continue
                for batch_d in range(bucket[0], bucket[1] + 1):
                    self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency'] / self.layers
                    G, delta = self.find_partition()
                    # print(f"batch_p: {batch_p}, batch_d: {batch_d}, G: {G}, delta: {delta}")
                    
                    if not self.check_slo(batch_p, batch_d, delta):
                        continue
                    
                    cost = self.cost(batch_p, batch_d)
                    if cost < best_cost:
                        best_batch_d = batch_d
                        best_batch_p = batch_p
                        best_G = deepcopy(G)
                        best_delta = delta
                        best_cost = cost
                break
        print(f"best batch_p: {best_batch_p}, best batch_d: {best_batch_d}, partition: {best_G}, with delta: {best_delta}")
        print(f"dfs count: {self.dfs_count}, original dfs count: {self.dfs_original_count}")
        prefill_latency = self.analysis.inference(best_batch_p, self.max_chunk_size)['prefill_latency']
        decode_latency = self.analysis.inference(best_batch_d, 1)['decode_latency']
        print(f"prefill latency: {prefill_latency}, decode latency: {decode_latency}")
        for i, layer in enumerate(best_G):
            print(f"Stage {i}: {layer} layers, prefill latency: {prefill_latency * best_G[i] / self.layers:.3f}, decode latency: {decode_latency * best_G[i] / self.layers:.3f}")

    def max_batch(self):
        # 二分查找 batch_p 的最大可行值
        left_p, right_p = 1, 99
        best_batch_p = 1
        while left_p <= right_p:
            mid_p = (left_p + right_p) // 2
            self.max_layer_per_gpu = self.max_layer_per_GPU(mid_p, self.max_chunk_size)
            self.max_stage = self.layers // self.max_layer_per_gpu if self.layers % self.max_layer_per_gpu == 0 else self.layers // self.max_layer_per_gpu + 1
            self.transfer_p = self.transfer_time(mid_p, self.max_chunk_size)
            prefill = self.analysis.inference(mid_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * self.transfer_p
            if prefill <= self.slo_p:
                best_batch_p = mid_p
                left_p = mid_p + 1
            else:
                right_p = mid_p - 1
        max_batch_p = best_batch_p

        # 二分查找 batch_d 的最大可行值
        left_d, right_d = 1, 999
        best_batch_d = 1
        while left_d <= right_d:
            mid_d = (left_d + right_d) // 2
            self.transfer_d = self.transfer_time(mid_d, 1)
            decode = self.analysis.inference(mid_d, 1)['decode_latency'] + (self.max_stage - 1) * self.transfer_d
            if decode <= self.slo_d:
                best_batch_d = mid_d
                left_d = mid_d + 1
            else:
                right_d = mid_d - 1
        max_batch_d = best_batch_d
        print(f"二分查找得到最大 batch_p: {max_batch_p}, batch_d: {max_batch_d}")
        return max_batch_p, max_batch_d

    def check_slo(self, batch_p, batch_d, delta):
        if self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency'] + (self.max_stage - 1) * (self.transfer_p + delta) > self.slo_p:
            return False
        if self.analysis.inference(batch_d, 1)['decode_latency'] + (self.max_stage - 1) * (self.transfer_d + delta) > self.slo_d:
            return False
        return True
    
    def cost(self, batch_p, batch_d, outnum=129):
        return (self.single_layer_prefill_latency / batch_p + outnum * self.single_layer_decode_latency / batch_d) * self.layers

    def max_layer_per_GPU(self,
                          batch_size,
                          seq_len,
                          memory_buffer_gb: float = 5.0):
        """_summary_

        Args:
            batch_size (_type_): _description_
            seq_len (_type_): _description_
            memory_buffer_gb (float, optional): _description_. Defaults to 2.0.

        Returns:
            _type_: 返回每个GPU上可以分配的最大层数
        """
        import math
        # print(model_config)
        weight_memory_per_layer = self.analysis.get_weight_memory_per_layer()
        memory_kv_cache_per_layer = self.analysis.get_memory_kv_cache_per_layer(
            batch_size=batch_size, seq_len=seq_len, kv_cache_dtype_bytes=2)
        activation_memory_per_layer = self.analysis.get_activation_memory_per_layer(
            batch_size=batch_size,
            seq_len=seq_len,
            layernorm_dtype_bytes=2,
            is_inference=True)

        weight_per_layer = weight_memory_per_layer + memory_kv_cache_per_layer + activation_memory_per_layer
        # print(weight_memory_per_layer / 1024 / 1024, memory_kv_cache_per_layer / 1024 / 1024, activation_memory_per_layer / 1024 / 1024)

        max_layer = math.floor(
            (self.gpu.mem_per_GPU_in_GB - memory_buffer_gb) /
            (weight_per_layer / 1024 / 1024 / 1024))
        return max_layer

    def transfer_time(self, batch_size, seq_len=1):
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
        # prefill = self.single_layer_prefill_latency = self.analysis.inference(batch_p, self.max_chunk_size)['prefill_latency']
        # decode = self.single_layer_decode_latency = self.analysis.inference(batch_d, 1)['decode_latency']
        # print(f"prefill latency: {prefill}, decode latency: {decode}")
        prefill_latencys = []
        decode_latencys = []
        for i in range(1, batch_p + 1):
            prefill = self.analysis.inference(i, self.max_chunk_size)['prefill_latency']
            prefill_latencys.append(prefill)
            # print(f"batch_p: {i}, prefill latency: {prefill}")
        for i in range(1, batch_d + 1, 20):
            decode = self.analysis.inference(i, 1)['decode_latency']
            decode_latencys.append(decode)
            # print(f"batch_d: {i}, decode latency: {decode}")
        print(f"prefill latency: {prefill_latencys},\n decode latency: {decode_latencys}")


if __name__ == "__main__":
    model = "Llama-2-7b"
    # model = "LLaMA-MoE-v1-3_0B-2_16"
    print(f"model: {model}")
    model_name = MODEL_CONFIG_FILE + model
    gpu_name = "t4-pcie-16gb"
    # gpu_name = "rtx6000-48gb"
    dtype_name = "w16a16e16"
    seq_len = 1024
    batch_size = 2

    gpu = load_gpus_from_config(gpu_name)

    model_config = get_model_config_by_name(model_name)
    # print(model_config)
    # {'name': '_workspace_gyk_model_config_Llama-2-7b', 'num_layers': 32, 'n_head': 32, 'hidden_dim': 4096, 'vocab_size': 32000, 'max_seq_len': 4096, 'num_key_value_heads': 32, 'num_key_value_groups': 1.0, 'ffn_embed_dim': 16384, 'expansion_ratio': 4, 'model_type': 'llama', 'moe_num_experts': 1, 'moe_top_k': 1}
    gpu_config = get_gpu_config_by_name(gpu_name)
    # GPUConfig(name='t4-pcie-16gb', mem_per_GPU_in_GB=160, hbm_bandwidth_in_GB_per_sec=300, intra_node_bandwidth_in_GB_per_sec=32, intra_node_min_message_latency=8e-06, peak_fp16_TFLOPS=65, peak_i8_TFLOPS=130, peak_i4_TFLOPS=260, inter_node_bandwidth_in_GB_per_sec=200)
    dtype_config = get_dtype_config_by_name(dtype_name)
    # DtypeConfig(name='w16a16e16', weight_bits=16, activation_bits=16, embedding_bits=16)
    parallel_config = ParallelismConfig(tp_size=1,
                                        pp_size=1,
                                        sp_size=1,
                                        dp_size=1)

    analysis = LLMAnalysis(
        model_config,
        gpu_config,
        dtype_config,
        parallel_config,
        flops_efficiency=gpu.flops_efficiency,
        hbm_memory_efficiency=gpu.hbm_memory_efficiency,
    )
    # print(analysis.model_config)
    # print(analysis.gpu_config)
    # print(analysis.dtype_config)

    # max_layer_per_GPU(model_config, gpu, dtype_name, seq_len, batch_size)
    # print(analysis.get_weight_memory_per_layer())
    # infer_data = analysis.inference(batch_size_per_gpu=2, seq_len=256)
    # print(f"prefill_latency: {infer_data['prefill_latency']}, decode_latency: {infer_data['decode_latency']}")
    # print(analysis.get_latency_fwd_per_layer(batch_size=50, seq_len=1, layernorm_dtype_bytes=2))
    # print(transfer_time(analysis, batch_size=2, seq_len=256))
    
    solver = solver(analysis, slo_p=1.5, slo_d=0.5, max_chunk_size=256)
    
    # solver.test(batch_p=8, batch_d=500)
    
    start_time = time.time()
    solver.solve_brute_force()
    print(f"brute force solve time: {time.time() - start_time:.2f} s")
    
    # solver.test()
    start_time = time.time()
    solver.solve_bucket()
    print(f"bucket solve time: {time.time() - start_time:.2f} s")
    
    
