import argparse
import gc
import json
import math
import os
from collections import defaultdict

import torch as t
from tqdm import tqdm

from attribution import patching_effect, jvp
from circuit_plotting import plot_circuit, plot_circuit_posaligned
from dictionary_learning import AutoEncoder
from data_loading_utils import load_examples, load_examples_nopair
from dictionary_loading_utils import load_saes_and_submodules, _get_preferred_device
from nnsight import LanguageModel
from coo_utils import sparse_reshape


def _safe_to_device(tensor, device):
    # Sparse tensors lack many ops on MPS; keep them on CPU instead.
    if isinstance(tensor, t.Tensor) and tensor.is_sparse and device.type == "mps":
        return tensor.to("cpu")
    return tensor.to(device)


def _sparse_sum_safe(sparse_tensor, dim):
    """Sum sparse tensor, handling MPS limitation by using CPU for sparse ops."""
    if sparse_tensor.is_sparse and sparse_tensor.device.type == "mps":
        sparse_tensor = sparse_tensor.to("cpu")
        result = sparse_tensor.sum(dim=dim)
        result = result.to("mps")
        return result
    return sparse_tensor.sum(dim=dim)


def _sparse_div_safe(sparse_tensor, scalar):
    """Divide sparse tensor by scalar, handling MPS limitation by using CPU for sparse ops."""
    if sparse_tensor.is_sparse and sparse_tensor.device.type == "mps":
        sparse_tensor = sparse_tensor.to("cpu")
        result = sparse_tensor / scalar
        result = result.to("mps")
        return result
    return sparse_tensor / scalar


def _sparse_reshape_safe(sparse_tensor, shape):
    """Reshape sparse tensor, handling MPS limitation by using CPU for sparse ops."""
    if sparse_tensor.is_sparse and sparse_tensor.device.type == "mps":
        sparse_tensor = sparse_tensor.to("cpu")
        result = sparse_reshape(sparse_tensor, shape)
        result = result.to("mps")
        return result
    return sparse_reshape(sparse_tensor, shape)


def _coerce_device(device_like):
    if device_like is None or isinstance(device_like, t.device):
        return device_like
    return t.device(device_like)


def _resolve_storage_device(requested_storage, compute_device):
    storage = _coerce_device(requested_storage)
    if storage is not None:
        return storage
    if compute_device.type == "cpu":
        return compute_device
    return t.device("cpu")


def _move_dictionaries_to(dictionaries, device):
    if device is None:
        return
    target = _coerce_device(device)
    seen = set()
    for dictionary in dictionaries.values():
        if dictionary is None:
            continue
        ident = id(dictionary)
        if ident in seen:
            continue
        seen.add(ident)
        dictionary.to(device=target)

def _use_tl_backend(model_name: str) -> bool:
    return model_name in {"gpt2", "openai-community/gpt2"}

def _tl_final_logits(logits: t.Tensor) -> t.Tensor:
    if logits.dim() == 3:
        return logits[:, -1, :]
    if logits.dim() == 2:
        return logits
    raise ValueError(f"Unexpected logits shape for TL backend: {tuple(logits.shape)}")

def get_circuit(
    clean,
    patch,
    model,
    embed,
    attns,
    mlps,
    resids,
    dictionaries,
    metric_fn,
    metric_kwargs=dict(),
    aggregation="sum",  # or "none" for not aggregating across sequence position
    nodes_only=False,
    parallel_attn=False,
    node_threshold=0.1,
    attrib_method="ig",
    patching_effect_fn=patching_effect,
    jvp_fn=jvp,
):
    all_submods = ([embed] if embed is not None else []) + [
        submod for layer_submods in zip(attns, mlps, resids) for submod in layer_submods
    ]

    # first get the patching effect of everything on y
    effects, deltas, grads, total_effect = patching_effect_fn(
        clean,
        patch,
        model,
        all_submods,
        dictionaries,
        metric_fn,
        metric_kwargs=metric_kwargs,
        method=attrib_method,
    )

    features_by_submod = {
        submod: effects[submod].abs() > node_threshold for submod in all_submods
    }

    n_layers = len(resids)

    nodes = {"y": total_effect}
    if embed is not None:
        nodes["embed"] = effects[embed]
    for i in range(n_layers):
        nodes[f"attn_{i}"] = effects[attns[i]]
        nodes[f"mlp_{i}"] = effects[mlps[i]]
        nodes[f"resid_{i}"] = effects[resids[i]]

    if nodes_only:
        if aggregation == "sum":
            for k in nodes:
                if k != "y":
                    nodes[k] = nodes[k].sum(dim=1)
        nodes = {k: v.mean(dim=0) for k, v in nodes.items()}
        return nodes, None

    edges = defaultdict(lambda: {})
    edges[f"resid_{len(resids) - 1}"] = {
        "y": effects[resids[-1]].to_tensor().flatten().to_sparse()
    }

    def N(upstream, downstream, midstream=[]):
        result = jvp_fn(
            clean,
            model,
            dictionaries,
            downstream,
            features_by_submod[downstream],
            upstream,
            grads[downstream],
            deltas[upstream],
            intermediate_stopgrads=midstream,
        )
        return result

    # now we work backward through the model to get the edges
    for layer in reversed(range(len(resids))):
        resid = resids[layer]
        mlp = mlps[layer]
        attn = attns[layer]

        MR_effect = N(mlp, resid)
        AR_effect = N(attn, resid, [mlp])
        edges[f"mlp_{layer}"][f"resid_{layer}"] = MR_effect
        edges[f"attn_{layer}"][f"resid_{layer}"] = AR_effect

        if not parallel_attn:
            AM_effect = N(attn, mlp)
            edges[f"attn_{layer}"][f"mlp_{layer}"] = AM_effect

        if layer > 0:
            prev_resid = resids[layer - 1]
        else:
            prev_resid = embed

        if prev_resid is not None:
            RM_effect = N(prev_resid, mlp, [attn])
            RA_effect = N(prev_resid, attn)
            RR_effect = N(prev_resid, resid, [mlp, attn])

            if layer > 0:
                edges[f"resid_{layer - 1}"][f"mlp_{layer}"] = RM_effect
                edges[f"resid_{layer - 1}"][f"attn_{layer}"] = RA_effect
                edges[f"resid_{layer - 1}"][f"resid_{layer}"] = RR_effect
            else:
                edges["embed"][f"mlp_{layer}"] = RM_effect
                edges["embed"][f"attn_{layer}"] = RA_effect
                edges["embed"]["resid_0"] = RR_effect

    # rearrange weight matrices
    for child in edges:
        # get shape for child
        bc, sc, fc = nodes[child].act.shape
        for parent in edges[child]:
            weight_matrix = edges[child][parent]
            if parent == "y":
                weight_matrix = sparse_reshape(weight_matrix, (bc, sc, fc + 1))
            else:
                continue
            edges[child][parent] = weight_matrix

    if aggregation == "sum":
        # aggregate across sequence position
        for child in edges:
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = _sparse_sum_safe(weight_matrix, dim=1)
                else:
                    weight_matrix = _sparse_sum_safe(weight_matrix, dim=(1, 4))
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].sum(dim=1)

        # aggregate across batch dimension
        for child in edges:
            bc, _ = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = _sparse_div_safe(_sparse_sum_safe(weight_matrix, dim=0), bc)
                else:
                    bp, _ = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = _sparse_div_safe(_sparse_sum_safe(weight_matrix, dim=(0, 2)), bc)
                edges[child][parent] = weight_matrix
        for node in nodes:
            if node != "y":
                nodes[node] = nodes[node].mean(dim=0)

    elif aggregation == "none":
        # aggregate across batch dimensions
        for child in edges:
            # get shape for child
            bc, sc, fc = nodes[child].act.shape
            for parent in edges[child]:
                weight_matrix = edges[child][parent]
                if parent == "y":
                    weight_matrix = _sparse_reshape_safe(weight_matrix, (bc, sc, fc + 1))
                    weight_matrix = _sparse_div_safe(_sparse_sum_safe(weight_matrix, dim=0), bc)
                else:
                    bp, sp, fp = nodes[parent].act.shape
                    assert bp == bc
                    weight_matrix = _sparse_div_safe(_sparse_sum_safe(weight_matrix, dim=(0, 3)), bc)
                edges[child][parent] = weight_matrix
        for node in nodes:
            nodes[node] = nodes[node].mean(dim=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return nodes, edges


def get_circuit_cluster(
    dataset,
    model_name="EleutherAI/pythia-70m-deduped",
    d_model=512,
    dict_id=10,
    dict_size=32768,
    max_length=64,
    max_examples=100,
    batch_size=2,
    node_threshold=0.1,
    edge_threshold=0.01,
    device="cuda:0",
    storage_device=None,
    dict_path="dictionaries/pythia-70m-deduped/",
    dataset_name="cluster_circuit",
    circuit_dir="circuits/",
    plot_dir="circuits/figures/",
    model=None,
    dictionaries=None,
):
    device = _coerce_device(device)
    storage_device = _resolve_storage_device(storage_device, device)
    model_configs = {
        "EleutherAI/pythia-70m-deduped": dict(
            layers=6,
            parallel_attn=True,
            include_embed=True,
            dtype=t.float32,
        ),
        "google/gemma-2-2b": dict(
            layers=26,
            parallel_attn=False,
            include_embed=False,
            dtype=t.bfloat16,
        ),
        "gpt2": dict(
            layers=12,
            parallel_attn=False,
            include_embed=True,
            dtype=t.float32,
        ),
        "openai-community/gpt2": dict(
            layers=12,
            parallel_attn=False,
            include_embed=True,
            dtype=t.float32,
        ),
    }
    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} not supported")
    config = model_configs[model_name]
    n_layers = config["layers"]
    parallel_attn = config["parallel_attn"]
    include_embed = config["include_embed"]
    dtype = config["dtype"]

    use_tl = _use_tl_backend(model_name)
    if model_name == "EleutherAI/pythia-70m-deduped":
        model = LanguageModel(model_name, device_map=device, dispatch=True, torch_dtype=dtype)
    elif model_name == "google/gemma-2-2b":
        model = LanguageModel(model_name, device_map=device, dispatch=True, attn_implementation="eager", torch_dtype=dtype)
    elif use_tl:
        from transformer_lens import HookedTransformer
        from tl_backend import load_tl_gpt2_saes_and_submodules

        model = HookedTransformer.from_pretrained(model_name, device=str(device))
    else:
        model = LanguageModel(model_name, device_map=device, dispatch=True, torch_dtype=dtype)

    if use_tl:
        from tl_backend import tl_jvp, tl_patching_effect

        submodules, dictionaries = load_tl_gpt2_saes_and_submodules(
            model,
            separate_by_type=True,
            include_embed=include_embed,
            device=device,
            thru_layer=None,
            neurons=False,
        )
        patching_effect_fn = tl_patching_effect
        jvp_fn = tl_jvp
    else:
        submodules, dictionaries = load_saes_and_submodules(
            model,
            separate_by_type=True,
            include_embed=include_embed,
            device=device,
            dtype=dtype,
        )
        patching_effect_fn = patching_effect
        jvp_fn = jvp
    if storage_device != device:
        _move_dictionaries_to(dictionaries, storage_device)
        dictionary_resident = storage_device
    else:
        dictionary_resident = device

    examples = load_examples_nopair(dataset, max_examples, model)
    num_examples = min(len(examples), max_examples)
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]
    if num_examples < max_examples:  # warn the user
        print(
            f"Total number of examples is less than {max_examples}. Using {num_examples} examples instead."
        )

    running_nodes = None
    running_edges = None

    for batch in tqdm(batches, desc="Batches"):
        if dictionary_resident != device:
            _move_dictionaries_to(dictionaries, device)
            dictionary_resident = device
        clean_inputs = [e["clean_prefix"] for e in batch]
        clean_answer_idxs = t.tensor(
            [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
            dtype=t.long,
            device=device
        )

        patch_inputs = None

        if use_tl:
            def metric_fn(logits):
                return -1 * t.gather(
                    _tl_final_logits(logits),
                    dim=-1,
                    index=clean_answer_idxs.view(-1, 1),
                ).squeeze(-1)
        else:
            def metric_fn(model):
                return -1 * t.gather(
                    model.output.logits[:, -1, :],
                    dim=-1,
                    index=clean_answer_idxs.view(-1, 1),
                ).squeeze(-1)

        nodes, edges = get_circuit(
            clean_inputs,
            patch_inputs,
            model,
            submodules.embed,
            submodules.attns,
            submodules.mlps,
            submodules.resids,
            dictionaries,
            metric_fn,
            aggregation="sum",
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            parallel_attn=parallel_attn,
            patching_effect_fn=patching_effect_fn,
            jvp_fn=jvp_fn,
        )
        if storage_device != device:
            _move_dictionaries_to(dictionaries, storage_device)
            dictionary_resident = storage_device

        if running_nodes is None:
            running_nodes = {
                k: len(batch) * nodes[k].to(storage_device) for k in nodes.keys() if k != "y"
            }
            running_edges = {
                k: {kk: len(batch) * edges[k][kk].to(storage_device) for kk in edges[k].keys()}
                for k in edges.keys()
            }
        else:
            for k in nodes.keys():
                if k != "y":
                    running_nodes[k] += len(batch) * nodes[k].to(storage_device)
            for k in edges.keys():
                for v in edges[k].keys():
                    running_edges[k][v] += len(batch) * edges[k][v].to(storage_device)

        # memory cleanup
        del nodes, edges
        gc.collect()

    nodes = {k: v.to(storage_device) / num_examples for k, v in running_nodes.items()}
    edges = {
        k: {
            kk: 1 / num_examples * _safe_to_device(v, storage_device)
            for kk, v in running_edges[k].items()
        }
        for k in running_edges.keys()
    }

    save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
    save_basename = f"{dataset_name}_dict{dict_id}_node{node_threshold}_edge{edge_threshold}_n{num_examples}_aggsum"
    with open(f"{circuit_dir}/{save_basename}.pt", "wb") as outfile:
        t.save(save_dict, outfile)

    nodes = save_dict["nodes"]
    edges = save_dict["edges"]

    annotations = None

    plot_circuit(
        nodes,
        edges,
        layers=n_layers,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        pen_thickness=1,
        annotations=annotations,
        save_dir=os.path.join(plot_dir, save_basename),
        gemma_mode=(model_name == "google/gemma-2-2b"),
        parallel_attn=parallel_attn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="simple_train",
        help="A subject-verb agreement dataset in data/, or a path to a cluster .json.",
    )
    parser.add_argument(
        "--num_examples",
        "-n",
        type=int,
        default=100,
        help="The number of examples from the --dataset over which to average indirect effects.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="The Huggingface ID of the model you wish to test.",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default="dictionaries/pythia-70m-deduped/",
        help="Path to all dictionaries for your language model.",
    )
    parser.add_argument(
        "--d_model", type=int, default=512, help="Hidden size of the language model."
    )
    parser.add_argument(
        "--use_neurons",
        default=False,
        action="store_true",
        help="Use neurons instead of features.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of examples to process at once when running circuit discovery.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum",
        help="Aggregation across token positions. Should be one of `sum` or `none`.",
    )
    parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.2,
        help="Indirect effect threshold for keeping circuit nodes.",
    )
    parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.02,
        help="Indirect effect threshold for keeping edges.",
    )
    parser.add_argument(
        "--thru_layer",
        type=int,
        default=None,
        help="Only load and analyze layers up to (and including) this index.",
    )
    parser.add_argument(
        "--attrib_method",
        type=str,
        default="ig",
        choices=["attrib", "ig", "exact"],
        help="Attribution method to use when computing node effects.",
    )
    parser.add_argument(
        "--pen_thickness",
        type=float,
        default=1,
        help="Scales the width of the edges in the circuit plot.",
    )
    parser.add_argument(
        "--nopair",
        default=False,
        action="store_true",
        help="Use if your data does not contain contrastive (minimal) pairs.",
    )
    parser.add_argument(
        "--plot_circuit",
        default=False,
        action="store_true",
        help="Plot the circuit after discovering it.",
    )
    parser.add_argument(
        "--nodes_only",
        default=False,
        action="store_true",
        help="Only search for causally implicated features; do not draw edges.",
    )
    parser.add_argument(
        "--plot_only",
        default=False,
        action="store_true",
        help="Do not run circuit discovery; just plot an existing circuit.",
    )
    parser.add_argument(
        "--circuit_dir",
        type=str,
        default="circuits",
        help="Directory to save/load circuits.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="circuits/figures/",
        help="Directory to save figures.",
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect MPS/CUDA/CPU)")
    parser.add_argument(
        "--storage_device",
        type=str,
        default=None,
        help="Device to keep dictionaries/aggregates on (defaults to CPU when using CUDA/MPS).",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = t.device(args.device)
    else:
        device = _get_preferred_device()

    print(f"Using device: {device}")
    storage_device = _resolve_storage_device(args.storage_device, device)
    print(f"Caching dictionaries/aggregates on: {storage_device}")

    model_configs = {
        "EleutherAI/pythia-70m-deduped": dict(
            layers=6,
            parallel_attn=True,
            include_embed=True,
            dtype=t.float32,
        ),
        "google/gemma-2-2b": dict(
            layers=26,
            parallel_attn=False,
            include_embed=False,
            dtype=t.bfloat16,
        ),
        "gpt2": dict(
            layers=12,
            parallel_attn=False,
            include_embed=True,
            dtype=t.float32,
        ),
        "openai-community/gpt2": dict(
            layers=12,
            parallel_attn=False,
            include_embed=True,
            dtype=t.float32,
        ),
    }
    if args.model not in model_configs:
        raise ValueError(f"Model {args.model} not supported")
    config = model_configs[args.model]
    if args.thru_layer is not None:
        n_layers = min(config["layers"], args.thru_layer + 1)
    else:
        n_layers = config["layers"]
    parallel_attn = config["parallel_attn"]
    include_embed = config["include_embed"]
    dtype = config["dtype"]

    use_tl = _use_tl_backend(args.model)
    if args.model == "EleutherAI/pythia-70m-deduped":
        model = LanguageModel(args.model, device_map=device, dispatch=True, torch_dtype=dtype)
    elif args.model == "google/gemma-2-2b":
        model = LanguageModel(
            args.model,
            device_map=device,
            dispatch=True,
            attn_implementation="eager",
            torch_dtype=dtype,
        )
    elif use_tl:
        from transformer_lens import HookedTransformer

        # HookedTransformer doesn't recognize openai-community/gpt2, map it to gpt2
        tl_model_name = args.model if args.model != "openai-community/gpt2" else "gpt2"
        model = HookedTransformer.from_pretrained(tl_model_name, device=str(device))
    else:  # GPT-2 variants (nnsight)
        model = LanguageModel(
            args.model,
            device_map=device,
            dispatch=True,
            torch_dtype=dtype,
        )

    if args.nopair:
        data_path = f"data/{args.dataset}.json"
        examples = load_examples_nopair(
            data_path, args.num_examples, model
        )
    else:
        data_path = f"data/{args.dataset}.json"
        examples = load_examples(
            data_path, args.num_examples, model, use_min_length_only=True
        )

    num_examples = min([args.num_examples, len(examples)])
    if num_examples < args.num_examples:  # warn the user
        print(
            f"Total number of examples is less than {args.num_examples}. Using {num_examples} examples instead."
        )

    batch_size = args.batch_size
    n_batches = math.ceil(num_examples / batch_size)
    batches = [
        examples[batch * batch_size : (batch + 1) * batch_size]
        for batch in range(n_batches)
    ]

    loaded_from_disk = False
    base_parts = [
        args.model.split("/")[-1],
        args.dataset,
        f"n{num_examples}",
        f"agg{args.aggregation}",
        f"attrib{args.attrib_method}",
    ]
    if args.thru_layer is not None:
        base_parts.append(f"thru{args.thru_layer}")
    if args.use_neurons:
        base_parts.append("neurons")
    if args.nodes_only:
        base_parts.append("nodesonly")
    save_stem = "_".join(base_parts + [f"node{args.node_threshold}", f"edge{args.edge_threshold}"])
    save_path = f"{args.circuit_dir}/{save_stem}.pt"
    if os.path.exists(save_path):
        print(f"Loading circuit from {save_path}")
        with open(save_path, "rb") as infile:
            save_dict = t.load(infile, weights_only=False)
        nodes = save_dict["nodes"]
        edges = save_dict["edges"]
        loaded_from_disk = True

    if not loaded_from_disk:
        print("computing circuit")
        if use_tl:
            from tl_backend import load_tl_gpt2_saes_and_submodules, tl_jvp, tl_patching_effect

            submodules, dictionaries = load_tl_gpt2_saes_and_submodules(
                model,
                separate_by_type=True,
                include_embed=include_embed,
                neurons=args.use_neurons,
                device=device,
                thru_layer=args.thru_layer,
            )
            patching_effect_fn = tl_patching_effect
            jvp_fn = tl_jvp
        else:
            submodules, dictionaries = load_saes_and_submodules(
                model,
                separate_by_type=True,
                include_embed=include_embed,
                neurons=args.use_neurons,
                device=device,
                dtype=dtype,
                thru_layer=args.thru_layer,
            )
            patching_effect_fn = patching_effect
            jvp_fn = jvp
        if storage_device != device:
            _move_dictionaries_to(dictionaries, storage_device)
            dictionary_resident = storage_device
        else:
            dictionary_resident = device

        running_nodes = None
        running_edges = None

        for batch in tqdm(batches, desc="Batches"):
            if dictionary_resident != device:
                _move_dictionaries_to(dictionaries, device)
                dictionary_resident = device
            clean_inputs = [e["clean_prefix"] for e in batch]
            clean_answer_idxs = t.tensor(
                [model.tokenizer(e["clean_answer"]).input_ids[-1] for e in batch],
                dtype=t.long,
                device=device,
            )

            if args.nopair:
                patch_inputs = None

                if use_tl:
                    def metric_fn(logits):
                        return -1 * t.gather(
                            _tl_final_logits(logits),
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                else:
                    def metric_fn(model):
                        return -1 * t.gather(
                            model.output.logits[:, -1, :],
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
            else:
                patch_inputs = [e["patch_prefix"] for e in batch]
                patch_answer_idxs = t.tensor(
                    [model.tokenizer(e["patch_answer"]).input_ids[-1] for e in batch],
                    dtype=t.long,
                    device=device,
                )

                if use_tl:
                    def metric_fn(logits):
                        return t.gather(
                            _tl_final_logits(logits),
                            dim=-1,
                            index=patch_answer_idxs.view(-1, 1),
                        ).squeeze(-1) - t.gather(
                            _tl_final_logits(logits),
                            dim=-1,
                            index=clean_answer_idxs.view(-1, 1),
                        ).squeeze(-1)
                else:
                    def metric_fn(model):
                        logits = model.output.logits[:, -1, :]
                        return t.gather(
                            logits, dim=-1, index=patch_answer_idxs.view(-1, 1)
                        ).squeeze(-1) - t.gather(
                            logits, dim=-1, index=clean_answer_idxs.view(-1, 1)
                        ).squeeze(-1)

            nodes, edges = get_circuit(
                clean_inputs,
                patch_inputs,
                model,
                submodules.embed,
                submodules.attns,
                submodules.mlps,
                submodules.resids,
                dictionaries,
                metric_fn,
                nodes_only=args.nodes_only,
                aggregation=args.aggregation,
                node_threshold=args.node_threshold,
                parallel_attn=parallel_attn,
                attrib_method=args.attrib_method,
                patching_effect_fn=patching_effect_fn,
                jvp_fn=jvp_fn,
            )
            if storage_device != device:
                _move_dictionaries_to(dictionaries, storage_device)
                dictionary_resident = storage_device

            if running_nodes is None:
                running_nodes = {
                    k: len(batch) * nodes[k].to(storage_device) for k in nodes.keys() if k != "y"
                }
                if not args.nodes_only:
                    running_edges = {
                        k: {
                            kk: len(batch) * edges[k][kk].to(storage_device)
                            for kk in edges[k].keys()
                        }
                        for k in edges.keys()
                    }
            else:
                for k in nodes.keys():
                    if k != "y":
                        running_nodes[k] += len(batch) * nodes[k].to(storage_device)
                if not args.nodes_only:
                    for k in edges.keys():
                        for v in edges[k].keys():
                            running_edges[k][v] += len(batch) * edges[k][v].to(storage_device)

            # memory cleanup
            del nodes, edges
            gc.collect()

        nodes = {k: v.to(storage_device) / num_examples for k, v in running_nodes.items()}
        if not args.nodes_only:
            edges = {
                k: {
                    kk: 1 / num_examples * _safe_to_device(v, storage_device)
                    for kk, v in running_edges[k].items()
                }
                for k in running_edges.keys()
            }
        else:
            edges = None

        save_dict = {"examples": examples, "nodes": nodes, "edges": edges}
        with open(save_path, "wb") as outfile:
            t.save(save_dict, outfile)

    # feature annotations
    if os.path.exists(
        annotations_path := f"annotations/{args.model.split('/')[-1]}.jsonl"
    ):
        print(f"Loading feature annotations from {annotations_path}")
        annotations = {}
        with open(annotations_path, "r") as f:
            for line in f:
                line = json.loads(line)
                if "Annotation" in line:
                    annotations[line["Name"]] = line["Annotation"]
    else:
        annotations = None

    if args.aggregation == "none":
        example = examples[0]["clean_prefix"]
        plot_circuit_posaligned(
            nodes,
            edges,
            layers=n_layers,
            example_text=example,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            pen_thickness=args.pen_thickness,
            annotations=annotations,
            save_dir=f"{args.plot_dir}/{save_stem}",
            gemma_mode=(args.model == "google/gemma-2-2b"),
            parallel_attn=parallel_attn,
        )
    else:
        plot_circuit(
            nodes,
            edges,
            layers=n_layers,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
            pen_thickness=args.pen_thickness,
            annotations=annotations,
            save_dir=f"{args.plot_dir}/{save_stem}",
            gemma_mode=(args.model == "google/gemma-2-2b"),
            parallel_attn=parallel_attn,
        )
