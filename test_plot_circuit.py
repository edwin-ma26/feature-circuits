import torch as t
from circuit_plotting import plot_circuit

path = "circuits/gpt2_simple_train_n5_aggsum_attribig_node0.2_edge0.02.pt"
print(f"Loading {path}")
data = t.load(path, map_location="cpu", weights_only=False)
print("Keys:", list(data.keys()))
nodes = data["nodes"]
edges = data.get("edges", None)

# Patch plot_circuit to skip rendering and just trigger the indexing logic
import circuit_plotting
original_render = circuit_plotting.Digraph.render

def no_render(self, *args, **kwargs):
    pass

circuit_plotting.Digraph.render = no_render

print("Calling plot_circuit...")
try:
    plot_circuit(nodes, edges, layers=6, node_threshold=0.1, edge_threshold=0.01, save_dir="test_out/circuit_test", gemma_mode=False, parallel_attn=True)
    print("Done - no IndexError!")
except IndexError as e:
    print(f"IndexError caught: {e}")
    import traceback
    traceback.print_exc()
