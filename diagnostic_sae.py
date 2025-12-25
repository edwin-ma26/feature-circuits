import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

def rel_err(x, x_hat, eps=1e-8):
    return (x - x_hat).norm() / (x.norm() + eps)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained("gpt2", device=device)
print("Loaded HookedTransformer")

# Load SAEs for layer 0
print("Loading SAEs...")
sae_resid = SAE.from_pretrained(release="gpt2-small-resid-post-v5-32k", sae_id="blocks.0.hook_resid_post", device=device)[0]
sae_attn  = SAE.from_pretrained(release="gpt2-small-attn-out-v5-32k",  sae_id="blocks.0.hook_attn_out", device=device)[0]
sae_mlp   = SAE.from_pretrained(release="gpt2-small-mlp-out-v5-32k",   sae_id="blocks.0.hook_mlp_out", device=device)[0]
print("SAEs loaded successfully")

prompt = "The friends that the dancer visits"
tokens = model.to_tokens(prompt)
print(f"Prompt tokens shape: {tokens.shape}")

cache = {}
def save(name):
    def hook(t, hook):
        cache[name] = t.detach()
    return hook

print("Running model with hooks...")
with torch.no_grad():
    _ = model.run_with_hooks(
        tokens,
        fwd_hooks=[
            ("blocks.0.hook_resid_post", save("resid_post")),
            ("blocks.0.hook_attn_out",   save("attn_out")),
            ("blocks.0.hook_mlp_out",    save("mlp_out")),
        ],
    )

x_r = cache["resid_post"]
x_a = cache["attn_out"]
x_m = cache["mlp_out"]

print(f"\nActivation shapes:")
print(f"  resid_post: {x_r.shape}")
print(f"  attn_out:   {x_a.shape}")
print(f"  mlp_out:    {x_m.shape}")

print("\nRunning SAEs...")
with torch.no_grad():
    x_r_hat = sae_resid(x_r)
    x_a_hat = sae_attn(x_a)
    x_m_hat = sae_mlp(x_m)

print("\n=== RECONSTRUCTION RESULTS ===")
print(f"resid_post: x norm {x_r.norm().item():.4f} | xhat norm {x_r_hat.norm().item():.4f} | rel_err {rel_err(x_r, x_r_hat).item():.6f}")
print(f"attn_out  : x norm {x_a.norm().item():.4f} | xhat norm {x_a_hat.norm().item():.4f} | rel_err {rel_err(x_a, x_a_hat).item():.6f}")
print(f"mlp_out   : x norm {x_m.norm().item():.4f} | xhat norm {x_m_hat.norm().item():.4f} | rel_err {rel_err(x_m, x_m_hat).item():.6f}")
print("\nExpected rel_err range: 0.05 to 0.8")
