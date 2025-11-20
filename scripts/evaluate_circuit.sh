#!/bin/bash

MODEL=$1
CIRCUIT=$2
EVAL_DATA=$3
THRESHOLD=$4
START_LAYER=$5
END_LAYER=${6:-}

CMD=(
python ablation.py
--model "$MODEL"
--circuit "$CIRCUIT"
--data "${EVAL_DATA}.json"
--num_examples 40
--dict_id "$DICTID"
--threshold "$THRESHOLD"
--ablation mean
--handle_errors default
--start_layer "$START_LAYER"
)

if [ -n "$END_LAYER" ]; then
  CMD+=(--end_layer "$END_LAYER")
fi

# Detect best available device (prefer MPS, then CUDA, else CPU)
DEVICE=$(python3 - <<'PY'
import torch
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
elif torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")
PY
)

CMD+=(
--batch_size 20
--device "$DEVICE"
)

"${CMD[@]}"
