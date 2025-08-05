
from copy import deepcopy


class ModelPartitioner:
    def __init__(self, analysis, max_stage=None, max_layer_per_gpu=None):
        self.analysis = analysis
        self.layers = analysis.model_config.num_layers
        self.max_stage = max_stage if max_stage else 1
        self.max_layer_per_gpu = max_layer_per_gpu if max_layer_per_gpu else self.layers

        # Initialize required attributes for dfs methods
        self.single_layer_prefill_latency = 0
        self.single_layer_decode_latency = 0
        self.transfer_p = 0
        self.transfer_d = 0

        # return 
        self.G = []  # Partition result
        self.delta = float('inf')  # Minimum difference
        
        # Add counters
        self.dfs_count = 0
        self.dfs_original_count = 0

    def BBSearch(self, G, stage, layers, max_diff, prev):
        """branch-and-bound for layer partitioning

        Args:
            G (list): Partition result
            stage (_type_): Current stage
            layers (_type_): Number of layers already allocated
            max_diff (_type_): Maximum prefill and decode latency difference from previous stages
            prev (_type_): Number of layers in the previous stage
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
            self.BBSearch(G_, stage + 1, layers + layer_now, new_diff, layer_now)

    def BBS_original(self, G, stage, layers, max_diff, prev):
        """Original branch-and-bound for layer partitioning

        Args:
            G (list): Partition result
            stage (_type_): Current stage
            layers (_type_): Number of layers already allocated
            max_diff (_type_): Maximum prefill and decode latency difference from previous stages
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
            self.BBS_original(G_, stage + 1, layers + layer_now, new_diff, layer_now)

    def find_partition(self):
        """Find the optimal partition for the model layers
        
        Returns:
            tuple: (G, delta) where G is the partition result and delta is the minimum difference
        """
        self.G = []
        self.delta = 100000
        
        upper = self.layers - (self.max_stage - 1)
        self.BBS_original(self.G, 0, 0, 0, upper)
        
        return self.G, self.delta