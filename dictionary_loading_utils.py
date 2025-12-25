from collections import namedtuple
from dictionary_learning import AutoEncoder, JumpReluAutoEncoder
from dictionary_learning.dictionary import IdentityDict
from attribution import Submodule
from typing import Literal
import torch as t
from huggingface_hub import list_repo_files
from tqdm import tqdm
import os

DICT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/dictionaries"


def _resolve_dict_subdir(base_dir: str, dict_id: str | int | None):
    """Return the dictionary subdirectory that matches dict_id."""
    available = sorted(
        entry
        for entry in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, entry))
    )
    if not available:
        raise FileNotFoundError(f"No dictionaries found under {base_dir}")
    if dict_id is None:
        return available[0]

    dict_id_str = str(dict_id)
    if dict_id_str in available:
        return dict_id_str
    matches = [entry for entry in available if entry.startswith(f"{dict_id_str}_")]
    if matches:
        return matches[0]
    raise ValueError(
        f"Dictionary ID {dict_id} not found under {base_dir}. Available: {', '.join(available)}"
    )


def _get_preferred_device() -> t.device:
    """Return the preferred torch device.

    Preference order: MPS (Apple Metal) if available, then CUDA, then CPU.
    """
    try:
        if hasattr(t.backends, "mps") and t.backends.mps.is_available():
            return t.device("mps")
    except Exception:
        # In case torch doesn't expose MPS backend the check can fail; ignore.
        pass
    if t.cuda.is_available():
        return t.device("cuda")
    return t.device("cpu")


def _safe_t_load(path: str):
    """Load a torch file mapping storages to CPU first to avoid CUDA-only deserialization errors."""
    try:
        return t.load(path, map_location="cpu")
    except Exception:
        # Fallback to default load if map_location fails for some reason
        return t.load(path)


def _safe_autoencoder_from_pretrained(path: str, dtype: t.dtype, device: t.device | None):
    state_dict = _safe_t_load(path)
    dict_size, activation_dim = state_dict["encoder.weight"].shape
    ae = AutoEncoder(activation_dim, dict_size)
    ae.load_state_dict(state_dict)
    if hasattr(ae, "normalize_decoder"):
        ae.normalize_decoder()
    if device is not None:
        ae = ae.to(dtype=dtype, device=device)
    return ae


def _safe_jumprelu_from_pretrained(
    path: str | None = None,
    load_from_sae_lens: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
    **kwargs,
):
    if load_from_sae_lens:
        # Let the library handle SAE Lens loading since it may not be a simple state dict
        return JumpReluAutoEncoder.from_pretrained(
            path, load_from_sae_lens=load_from_sae_lens, dtype=dtype, device=device, **kwargs
        )
    state_dict = _safe_t_load(path)
    activation_dim, dict_size = state_dict["W_enc"].shape
    ae = JumpReluAutoEncoder(activation_dim, dict_size)
    ae.load_state_dict(state_dict)
    ae = ae.to(dtype=dtype, device=device)
    return ae

DictionaryStash = namedtuple("DictionaryStash", ["embed", "attns", "mlps", "resids"])


GPT2_RELEASES = {
    "attn": "jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs",
    "mlp": "jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs",
    "resid": "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
}


def _load_pythia_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
    dict_id: str | int | None = "10_32768",
):
    if device is None:
        device = _get_preferred_device()
    assert (
        len(model.gpt_neox.layers) == 6
    ), "Not the expected number of layers for pythia-70m-deduped"
    if thru_layer is None:
        thru_layer = len(model.gpt_neox.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    dict_id_value: str | int | None = dict_id

    def _dict_path(base_dir: str) -> str:
        subdir = _resolve_dict_subdir(base_dir, dict_id_value)
        return os.path.join(base_dir, subdir, "ae.pt")

    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.gpt_neox.embed_in,
        )
        if not neurons:
            dictionaries[embed] = _safe_autoencoder_from_pretrained(
                _dict_path(f"{DICT_DIR}/pythia-70m-deduped/embed"),
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[embed] = IdentityDict(512)
    else:
        embed = None
    for i, layer in enumerate(model.gpt_neox.layers[: thru_layer + 1]):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.attention,
                is_tuple=True,
            )
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        if not neurons:
            dictionaries[attn] = _safe_autoencoder_from_pretrained(
                _dict_path(f"{DICT_DIR}/pythia-70m-deduped/attn_out_layer{i}"),
                dtype=dtype,
                device=device,
            )
            dictionaries[mlp] = _safe_autoencoder_from_pretrained(
                _dict_path(f"{DICT_DIR}/pythia-70m-deduped/mlp_out_layer{i}"),
                dtype=dtype,
                device=device,
            )
            dictionaries[resid] = _safe_autoencoder_from_pretrained(
                _dict_path(f"{DICT_DIR}/pythia-70m-deduped/resid_out_layer{i}"),
                dtype=dtype,
                device=device,
            )
        else:
            dictionaries[attn] = IdentityDict(512)
            dictionaries[mlp] = IdentityDict(512)
            dictionaries[resid] = IdentityDict(512)

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gpt2_sae(
    submod_type: Literal["attn", "mlp", "resid"],
    layer: int,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
):
    if device is None:
        device = _get_preferred_device()

    if neurons:
        return IdentityDict(768)

    if submod_type not in GPT2_RELEASES:
        raise ValueError(f"Unsupported GPT-2 submodule type: {submod_type}")

    repo_id = GPT2_RELEASES[submod_type]
    sae_layer = f"v5_32k_layer_{layer}"
    if submod_type == "resid":
        sae_layer += ".pt"

    return _safe_jumprelu_from_pretrained(
        None,
        load_from_sae_lens=True,
        release=repo_id,
        sae_id=sae_layer,
        dtype=dtype,
        device=device,
    )


def _load_gpt2_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
):
    if device is None:
        device = _get_preferred_device()
    assert (
        len(model.transformer.h) == 12
    ), "Not the expected number of layers for GPT-2 Small"
    if thru_layer is None:
        thru_layer = len(model.transformer.h)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.transformer.wte,
        )
        dictionaries[embed] = IdentityDict(768)
    else:
        embed = None

    for i, layer in enumerate(model.transformer.h[: thru_layer + 1]):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}",
                submodule=layer.attn,
                is_tuple=True,
            )
        )
        dictionaries[attn] = load_gpt2_sae(
            "attn", i, neurons=neurons, dtype=dtype, device=device
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.mlp,
            )
        )
        dictionaries[mlp] = load_gpt2_sae(
            "mlp", i, neurons=neurons, dtype=dtype, device=device
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gpt2_sae(
            "resid", i, neurons=neurons, dtype=dtype, device=device
        )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_gemma_sae(
    submod_type: Literal["embed", "attn", "mlp", "resid"],
    layer: int,
    width: Literal["16k", "65k"] = "16k",
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
):
    if device is None:
        device = _get_preferred_device()

    if neurons:
        if submod_type != "attn":
            return IdentityDict(2304)
        else:
            return IdentityDict(2048)

    repo_id = "google/gemma-scope-2b-pt-" + (
        "res"
        if submod_type in ["embed", "resid"]
        else "att" if submod_type == "attn" else "mlp"
    )
    if submod_type != "embed":
        directory_path = f"layer_{layer}/width_{width}"
    else:
        directory_path = "embedding/width_4k"

    files_with_l0s = [
        (f, int(f.split("_")[-1].split("/")[0]))
        for f in list_repo_files(repo_id, repo_type="model", revision="main")
        if f.startswith(directory_path) and f.endswith("params.npz")
    ]
    optimal_file = min(files_with_l0s, key=lambda x: abs(x[1] - 100))[0]
    optimal_file = optimal_file.split("/params.npz")[0]
    return _safe_jumprelu_from_pretrained(
        None,
        load_from_sae_lens=True,
        release=repo_id.split("google/")[-1],
        sae_id=optimal_file,
        dtype=dtype,
        device=device,
    )


def _load_gemma_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
):
    if device is None:
        device = _get_preferred_device()
    assert (
        len(model.model.layers) == 26
    ), "Not the expected number of layers for Gemma-2-2B"
    if thru_layer is None:
        thru_layer = len(model.model.layers)

    attns = []
    mlps = []
    resids = []
    dictionaries = {}
    if include_embed:
        embed = Submodule(
            name="embed",
            submodule=model.model.embed_tokens,
        )
        dictionaries[embed] = load_gemma_sae(
            "embed", 0, neurons=neurons, dtype=dtype, device=device
        )
    else:
        embed = None
    for i, layer in tqdm(
        enumerate(model.model.layers[: thru_layer + 1]),
        total=thru_layer + 1,
        desc="Loading Gemma SAEs",
    ):
        attns.append(
            attn := Submodule(
                name=f"attn_{i}", submodule=layer.self_attn.o_proj, use_input=True
            )
        )
        dictionaries[attn] = load_gemma_sae(
            "attn", i, neurons=neurons, dtype=dtype, device=device
        )
        mlps.append(
            mlp := Submodule(
                name=f"mlp_{i}",
                submodule=layer.post_feedforward_layernorm,
            )
        )
        dictionaries[mlp] = load_gemma_sae(
            "mlp", i, neurons=neurons, dtype=dtype, device=device
        )
        resids.append(
            resid := Submodule(
                name=f"resid_{i}",
                submodule=layer,
                is_tuple=True,
            )
        )
        dictionaries[resid] = load_gemma_sae(
            "resid", i, neurons=neurons, dtype=dtype, device=device
        )

    if separate_by_type:
        return DictionaryStash(embed, attns, mlps, resids), dictionaries
    else:
        submodules = ([embed] if include_embed else []) + [
            x
            for layer_dictionaries in zip(attns, mlps, resids)
            for x in layer_dictionaries
        ]
        return submodules, dictionaries


def load_saes_and_submodules(
    model,
    thru_layer: int | None = None,
    separate_by_type: bool = False,
    include_embed: bool = True,
    neurons: bool = False,
    dtype: t.dtype = t.float32,
    device: t.device | None = None,
    dict_id: str | int | None = "10_32768",
):
    if device is None:
        device = _get_preferred_device()
    model_name = model.config._name_or_path

    if model_name == "EleutherAI/pythia-70m-deduped":
        return _load_pythia_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
            dict_id=dict_id,
        )
    elif model_name == "google/gemma-2-2b":
        return _load_gemma_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=include_embed,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    elif model_name in {"gpt2", "openai-community/gpt2"}:
        return _load_gpt2_saes_and_submodules(
            model,
            thru_layer=thru_layer,
            separate_by_type=separate_by_type,
            include_embed=False,
            neurons=neurons,
            dtype=dtype,
            device=device,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
