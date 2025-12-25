import time
import torch as t
from transformers import AutoModelForCausalLM
import dictionary_loading_utils as dlu

print('GPT2_RELEASES mapping:')
for k,v in dlu.GPT2_RELEASES.items():
    print(' ',k, '->', v)

print('\nExpected SAE IDs for layer 0:')
for sub in ['attn','mlp','resid']:
    if sub == 'attn':
        sid = f'blocks.0.hook_attn_out'
    elif sub == 'mlp':
        sid = f'blocks.0.hook_mlp_out'
    else:
        sid = f'blocks.0.hook_resid_post'
    print(f'  {sub}: release={dlu.GPT2_RELEASES[sub]}, sae_id={sid}')

# Load a small HF GPT-2 model (might download if not cached)
print('\nLoading HF GPT-2 model (this may download ~500MB if not cached)...')
model = AutoModelForCausalLM.from_pretrained('gpt2')
print('Model loaded; model.config._name_or_path =', model.config._name_or_path)

print('\nCalling load_saes_and_submodules (thru_layer=0, include_embed=False)')
start = time.time()
submods, dicts = dlu.load_saes_and_submodules(model, thru_layer=0, include_embed=False, neurons=False, device=t.device('cpu'))
print('Loaded in', time.time()-start, 's')

print('\nReturned submodules:')
for s in submods:
    print('  ', s.name)

print('\nDictionaries mapping:')
for k,v in dicts.items():
    # k is a Submodule object, print name and type of v
    try:
        name = k.name
    except Exception:
        name = str(k)
    print('  ', name, '->', type(v).__name__)

print('\nIf types are JumpReluAutoEncoder/SAE objects (not IdentityDict) then SAEs loaded.')
