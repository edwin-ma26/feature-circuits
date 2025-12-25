import numpy as np
from circuit_plotting import plot_circuit

class Dummy:
    def __init__(self, arr):
        self._arr = np.array(arr)
    def to_tensor(self):
        return self._arr

# Create nodes: upstream has length 769 with small values, some > threshold
up_len = 769
down_len = 10
upstream_vals = np.zeros(up_len)
# set some values above threshold
upstream_vals[0] = 0.2
upstream_vals[100] = 0.15
upstream_vals[768] = 0.12

# downstream values
downstream_vals = np.zeros(down_len)
downstream_vals[0] = 0.2

# weight matrix has upstream dimension 770 (> up_len) to simulate mismatch
weight_matrix = np.zeros((down_len, up_len + 1))
# set some weights above threshold
weight_matrix[0, 0] = 0.02
weight_matrix[0, 100] = 0.02
weight_matrix[0, 768] = 0.02
weight_matrix[0, 769] = 0.02  # this column has no corresponding upstream value

# Build nodes and edges dicts
nodes = {
    'attn_0': Dummy(upstream_vals),
    'resid_0': Dummy(downstream_vals),
    'resid_5': Dummy(np.zeros(5)),
    'y': Dummy(np.array([0.0]))
}

edges = {
    'attn_0': {
        'resid_0': weight_matrix
    }
}

print('Running plot_circuit with synthetic data...')
plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, save_dir='test_out/numpy_test', gemma_mode=False, parallel_attn=True)
print('plot_circuit finished without IndexError')
