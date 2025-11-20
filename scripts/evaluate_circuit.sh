#!/bin/bash
set -euo pipefail

if [ "$#" -lt 9 ]; then
  cat <<'USAGE' >&2
Usage: scripts/evaluate_circuit.sh MODEL TRAIN_DATA EVAL_DATA NUM_EXAMPLES AGGREGATION ATTRIB_METHOD NODE_THRESHOLD EDGE_THRESHOLD START_LAYER [END_LAYER] [options]

Options:
  --thru_layer N         Layer index used during discovery (to match filenames)
  --nodes_only           Circuit was generated with --nodes_only
  --use_neurons          Circuit was generated with --use_neurons
  --circuit_dir DIR      Directory containing saved circuits (default: circuits)
  --dict_id ID           Dictionary identifier for SAEs (default: $DICTID or 10_32768)
  --num_eval_examples N  Number of evaluation examples (default: 40)
  --batch_size N         Evaluation batch size (default: 20)
  --ablation MODE        Ablation style passed to ablation.py (default: mean)
  --handle_errors MODE   How to treat SAE residuals (default: default)
  --device DEV           Force evaluation device (auto-detected if omitted)
USAGE
  exit 1
fi

MODEL=$1
TRAIN_DATA=$2
EVAL_DATA=$3
DISCOVERY_EXAMPLES=$4
AGGREGATION=$5
ATTRIB=$6
NODE_THRESHOLD=$7
EDGE_THRESHOLD=$8
START_LAYER=$9
shift 9

END_LAYER=""
if [ "$#" -gt 0 ] && [[ $1 != --* ]]; then
  END_LAYER=$1
  shift
fi

THRU_LAYER=""
NODES_ONLY=0
USE_NEURONS=0
CIRCUIT_DIR="circuits"
DICT_ID=${DICTID:-10_32768}
EVAL_EXAMPLES=40
BATCH_SIZE=20
ABALATION_MODE="mean"
HANDLE_ERRORS="default"
FORCED_DEVICE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --thru_layer)
      THRU_LAYER=$2; shift 2 ;;
    --nodes_only)
      NODES_ONLY=1; shift ;;
    --use_neurons)
      USE_NEURONS=1; shift ;;
    --circuit_dir)
      CIRCUIT_DIR=$2; shift 2 ;;
    --dict_id)
      DICT_ID=$2; shift 2 ;;
    --num_eval_examples)
      EVAL_EXAMPLES=$2; shift 2 ;;
    --batch_size)
      BATCH_SIZE=$2; shift 2 ;;
    --ablation)
      ABALATION_MODE=$2; shift 2 ;;
    --handle_errors)
      HANDLE_ERRORS=$2; shift 2 ;;
    --device)
      FORCED_DEVICE=$2; shift 2 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1 ;;
  esac
done

MODEL_NAME=${MODEL##*/}
CIRCUIT_STEM="${MODEL_NAME}_${TRAIN_DATA}_n${DISCOVERY_EXAMPLES}_agg${AGGREGATION}_attrib${ATTRIB}"
if [ -n "$THRU_LAYER" ]; then
  CIRCUIT_STEM="${CIRCUIT_STEM}_thru${THRU_LAYER}"
fi
if [ "$USE_NEURONS" -eq 1 ]; then
  CIRCUIT_STEM="${CIRCUIT_STEM}_neurons"
fi
if [ "$NODES_ONLY" -eq 1 ]; then
  CIRCUIT_STEM="${CIRCUIT_STEM}_nodesonly"
fi
CIRCUIT_STEM="${CIRCUIT_STEM}_node${NODE_THRESHOLD}_edge${EDGE_THRESHOLD}"
CIRCUIT_PATH="${CIRCUIT_DIR%/}/${CIRCUIT_STEM}.pt"

if [ ! -f "$CIRCUIT_PATH" ]; then
  echo "Circuit file not found: $CIRCUIT_PATH" >&2
  exit 1
fi

# Determine device if not forced
if [ -n "$FORCED_DEVICE" ]; then
  DEVICE="$FORCED_DEVICE"
else
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
fi

CMD=(
python ablation.py
--model "$MODEL"
--circuit "$CIRCUIT_PATH"
--data "$EVAL_DATA"
--num_examples "$EVAL_EXAMPLES"
--dict_id "$DICT_ID"
--threshold "$NODE_THRESHOLD"
--ablation "$ABALATION_MODE"
--handle_errors "$HANDLE_ERRORS"
--start_layer "$START_LAYER"
--batch_size "$BATCH_SIZE"
--device "$DEVICE"
)

if [ -n "$END_LAYER" ]; then
  CMD+=(--end_layer "$END_LAYER")
fi

"${CMD[@]}"
