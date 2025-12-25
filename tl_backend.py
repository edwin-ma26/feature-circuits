from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch as t
from sae_lens import SAE

from activation_utils import SparseAct
from dictionary_learning.dictionary import IdentityDict
from loading_utils import DictionaryStash


@dataclass(frozen=True)
class TLSubmodule:
    name: str
    hook_name: str

    def __hash__(self) -> int:
        return hash(self.name)


GPT2_SAE_RELEASES = {
    "attn": "gpt2-small-attn-out-v5-32k",
    "mlp": "gpt2-small-mlp-out-v5-32k",
    "resid": "gpt2-small-resid-post-v5-32k",
}


def load_tl_gpt2_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    device: t.device | None = None,
):
    n_layers = model.cfg.n_layers
    if thru_layer is None:
        thru_layer = n_layers - 1

    attns = []
    mlps = []
    resids = []
    dictionaries = {}

    if include_embed:
        embed = TLSubmodule(name="embed", hook_name="hook_embed")
        dictionaries[embed] = IdentityDict(768)
    else:
        embed = None

    for i in range(thru_layer + 1):
        attn = TLSubmodule(name=f"attn_{i}", hook_name=f"blocks.{i}.hook_attn_out")
        mlp = TLSubmodule(name=f"mlp_{i}", hook_name=f"blocks.{i}.hook_mlp_out")
        resid = TLSubmodule(name=f"resid_{i}", hook_name=f"blocks.{i}.hook_resid_post")

        attns.append(attn)
        mlps.append(mlp)
        resids.append(resid)

        if neurons:
            dictionaries[attn] = IdentityDict(768)
            dictionaries[mlp] = IdentityDict(768)
            dictionaries[resid] = IdentityDict(768)
        else:
            dictionaries[attn] = SAE.from_pretrained(
                release=GPT2_SAE_RELEASES["attn"],
                sae_id=attn.hook_name,
                device=str(device) if device is not None else "cpu",
            )
            dictionaries[mlp] = SAE.from_pretrained(
                release=GPT2_SAE_RELEASES["mlp"],
                sae_id=mlp.hook_name,
                device=str(device) if device is not None else "cpu",
            )
            dictionaries[resid] = SAE.from_pretrained(
                release=GPT2_SAE_RELEASES["resid"],
                sae_id=resid.hook_name,
                device=str(device) if device is not None else "cpu",
            )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    submodules = ([embed] if include_embed else []) + [
        x for layer_dictionaries in zip(attns, mlps, resids) for x in layer_dictionaries
    ]
    return submodules, dictionaries


def _dict_forward(dictionary, x, output_features: bool = False):
    if isinstance(dictionary, SAE):
        f = dictionary.encode(x)
        x_hat = dictionary.decode(f)
        return (x_hat, f) if output_features else x_hat
    if output_features:
        return dictionary(x, output_features=True)
    return dictionary(x)


def _dict_decode(dictionary, f):
    if isinstance(dictionary, SAE):
        return dictionary.decode(f)
    return dictionary.decode(f)


def _tl_collect_states(model, tokens, submodules, dictionaries, require_grad: bool):
    states = {}
    hooks = []

    def make_hook(submod):
        def hook(act, hook):  # noqa: ARG001
            x_hat, f = _dict_forward(dictionaries[submod], act, output_features=True)
            res = act - x_hat
            if require_grad:
                f.retain_grad()
                res.retain_grad()
            states[submod] = SparseAct(act=f, res=res)
            return act

        return hook

    for submod in submodules:
        hooks.append((submod.hook_name, make_hook(submod)))

    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    return logits, states


def _tl_pe_attrib(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn: Callable,
    metric_kwargs: dict | None = None,
):
    metric_kwargs = metric_kwargs or {}
    tokens = model.to_tokens(clean)
    logits, hidden_states_clean = _tl_collect_states(
        model, tokens, submodules, dictionaries, require_grad=True
    )
    metric_clean = metric_fn(logits, **metric_kwargs)
    metric_clean.sum().backward()

    grads = {k: v.grad for k, v in hidden_states_clean.items()}

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(
                act=t.zeros_like(v.act),
                res=t.zeros_like(v.res),
            )
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        with t.no_grad():
            patch_tokens = model.to_tokens(patch)
            logits_patch, hidden_states_patch = _tl_collect_states(
                model, patch_tokens, submodules, dictionaries, require_grad=False
            )
            metric_patch = metric_fn(logits_patch, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()

    effects = {}
    deltas = {}
    for submodule in submodules:
        patch_state = hidden_states_patch[submodule]
        clean_state = hidden_states_clean[submodule]
        grad = grads[submodule]
        delta = patch_state - clean_state.detach() if patch_state is not None else -clean_state.detach()
        effect = delta @ grad
        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return effects, deltas, grads, total_effect


def _tl_pe_ig(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn: Callable,
    steps: int = 10,
    metric_kwargs: dict | None = None,
):
    metric_kwargs = metric_kwargs or {}
    tokens = model.to_tokens(clean)
    with t.no_grad():
        logits_clean, hidden_states_clean = _tl_collect_states(
            model, tokens, submodules, dictionaries, require_grad=False
        )
        metric_clean = metric_fn(logits_clean, **metric_kwargs)

    if patch is None:
        hidden_states_patch = {
            k: SparseAct(
                act=t.zeros_like(v.act),
                res=t.zeros_like(v.res),
            )
            for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        with t.no_grad():
            patch_tokens = model.to_tokens(patch)
            logits_patch, hidden_states_patch = _tl_collect_states(
                model, patch_tokens, submodules, dictionaries, require_grad=False
            )
            metric_patch = metric_fn(logits_patch, **metric_kwargs)
        total_effect = (metric_patch - metric_clean).detach()

    fs_by_submod = {s: [] for s in submodules}
    metrics = []
    for step in range(steps):
        alpha = step / steps
        hooks = []
        for submod in submodules:
            clean_state = hidden_states_clean[submod]
            patch_state = hidden_states_patch[submod]
            f_act = (1 - alpha) * clean_state.act + alpha * patch_state.act
            f_res = (1 - alpha) * clean_state.res + alpha * patch_state.res
            f_act.requires_grad_().retain_grad()
            f_res.requires_grad_().retain_grad()
            fs_by_submod[submod].append(SparseAct(act=f_act, res=f_res))

            def hook(act, hook, submod=submod, f_act=f_act, f_res=f_res):  # noqa: ARG001
                x_hat = _dict_decode(dictionaries[submod], f_act)
                return x_hat + f_res

            hooks.append((submod.hook_name, hook))

        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        metrics.append(metric_fn(logits, **metric_kwargs))

    metric = sum(metrics)
    metric.sum().backward()

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        first_state = fs_by_submod[submodule][0]
        grad_act_total = t.zeros_like(first_state.act)
        grad_res_total = t.zeros_like(first_state.res)
        for f in fs_by_submod[submodule]:
            if f.act.grad is not None:
                grad_act_total = grad_act_total + f.act.grad
            if f.res.grad is not None:
                grad_res_total = grad_res_total + f.res.grad
        mean_grad_act = grad_act_total / steps
        mean_grad_res = grad_res_total / steps
        grad = SparseAct(act=mean_grad_act, res=mean_grad_res)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return effects, deltas, grads, total_effect


def tl_patching_effect(
    clean,
    patch,
    model,
    submodules,
    dictionaries,
    metric_fn: Callable,
    method: str = "attrib",
    steps: int = 10,
    metric_kwargs: dict | None = None,
):
    if method == "attrib":
        return _tl_pe_attrib(
            clean,
            patch,
            model,
            submodules,
            dictionaries,
            metric_fn,
            metric_kwargs=metric_kwargs,
        )
    if method == "ig":
        return _tl_pe_ig(
            clean,
            patch,
            model,
            submodules,
            dictionaries,
            metric_fn,
            steps=steps,
            metric_kwargs=metric_kwargs,
        )
    if method == "exact":
        raise NotImplementedError("Exact attribution is not supported for the TransformerLens backend.")
    raise ValueError(f"Unknown method {method}")


def tl_jvp(
    input,
    model,
    dictionaries,
    downstream_submod,
    downstream_features,
    upstream_submod,
    left_vec: SparseAct,
    right_vec: SparseAct,
    intermediate_stopgrads: list[TLSubmodule] | None = None,
):
    if intermediate_stopgrads is None:
        intermediate_stopgrads = []

    downstream_dict = dictionaries[downstream_submod]
    upstream_dict = dictionaries[upstream_submod]
    b, s, n_feats = downstream_features.act.shape

    if t.all(downstream_features.to_tensor() == 0):
        return t.sparse_coo_tensor(
            t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long),
            t.zeros(0),
            size=(b, s, n_feats + 1, b, s, n_feats + 1),
        ).to(model.cfg.device)

    tokens = model.to_tokens(input)
    states = {}
    stopgrads = {}
    hooks = []

    def make_hook(submod):
        def hook(act, hook):  # noqa: ARG001
            x_hat, f = _dict_forward(dictionaries[submod], act, output_features=True)
            res = act - x_hat
            f.retain_grad()
            res.retain_grad()
            states[submod] = SparseAct(act=f, res=res)
            if submod in intermediate_stopgrads:
                act.retain_grad()
                stopgrads[submod] = act
            return act

        return hook

    for submod in {downstream_submod, upstream_submod, *intermediate_stopgrads}:
        hooks.append((submod.hook_name, make_hook(submod)))

    _ = model.run_with_hooks(tokens, fwd_hooks=hooks)

    upstream_act = states[upstream_submod]
    downstream_act = states[downstream_submod]

    to_backprops = (left_vec @ downstream_act).to_tensor()
    downstream_idxs = downstream_features.to_tensor().nonzero()

    upstream_idxs = []
    values = []
    downstream_idx_list = []

    for downstream_idx in downstream_idxs:
        if upstream_act.act.grad is not None:
            upstream_act.act.grad.zero_()
        if upstream_act.res.grad is not None:
            upstream_act.res.grad.zero_()
        for submod in intermediate_stopgrads:
            act = stopgrads.get(submod)
            if act is not None and act.grad is not None:
                act.grad.zero_()

        to_backprops[tuple(downstream_idx)].backward(retain_graph=True)

        grad_act = upstream_act.act.grad
        grad_res = upstream_act.res.grad
        if grad_act is None:
            grad_act = t.zeros_like(upstream_act.act)
        if grad_res is None:
            grad_res = t.zeros_like(upstream_act.res)
        vjv = (SparseAct(act=grad_act, res=grad_res) @ right_vec).to_tensor()
        mask = vjv != 0
        coords = mask.nonzero()
        downstream_idx_list.append(downstream_idx.clone())
        upstream_idxs.append(coords)
        values.append(vjv[coords[:, 0], coords[:, 1], coords[:, 2]])

    idx_chunks = []
    value_chunks = []
    for downstream_idx, upstream_idx, vals in zip(
        downstream_idx_list, upstream_idxs, values
    ):
        if upstream_idx.numel() == 0 or vals.numel() == 0:
            continue
        if upstream_idx.shape[0] != vals.shape[0]:
            raise RuntimeError(
                f"Mismatch between upstream indices ({upstream_idx.shape[0]}) and values ({vals.shape[0]}) "
                f"for edge {downstream_submod.name}->{upstream_submod.name}"
            )
        repeated_downstream = downstream_idx.repeat(upstream_idx.shape[0], 1)
        idx_chunks.append(t.cat([repeated_downstream, upstream_idx], dim=1))
        value_chunks.append(vals)

    if len(value_chunks) == 0:
        return t.sparse_coo_tensor(
            t.zeros((2 * downstream_features.act.dim(), 0), dtype=t.long),
            t.zeros(0),
            size=(b, s, n_feats + 1, b, s, n_feats + 1),
        ).to(model.cfg.device)

    idx = t.cat(idx_chunks, dim=0).T
    val = t.cat(value_chunks, dim=0)
    return t.sparse_coo_tensor(
        idx, val, size=(b, s, n_feats + 1, b, s, n_feats + 1)
    ).to(model.cfg.device)
