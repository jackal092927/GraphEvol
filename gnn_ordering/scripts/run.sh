#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   bash scripts/run.sh --round 0 --tau_hat_source degree
#   bash scripts/run.sh --round 0 --tau_hat_source degree --overwrite
#   bash scripts/run.sh --round 1 --tau_hat_source file --tau_hat_path outputs/orderings/round0/tau_hat_hat_all.jsonl
#   bash scripts/run.sh --round 1 --tau_hat_source file --tau_hat_path outputs/orderings/round0/tau_hat_hat_all.jsonl --overwrite

OVERWRITE=0

ROUND=""
TAU_HAT_SOURCE="degree"
TAU_HAT_PATH=""
SPLIT_PATH=""

N_NODES=30
NUM_GRAPHS=120
M=2
ER=0.05
SEED=7

EPOCHS=200
LR=1e-3
WD=1e-4

ANNEAL_STEPS=3000
ANNEAL_T0=1.0
ANNEAL_TEND=0.001

while [[ $# -gt 0 ]]; do
  case "$1" in
    --round) ROUND="$2"; shift 2;;
    --tau_hat_source) TAU_HAT_SOURCE="$2"; shift 2;;
    --tau_hat_path) TAU_HAT_PATH="$2"; shift 2;;
    --split_path) SPLIT_PATH="$2"; shift 2;;

    --n_nodes) N_NODES="$2"; shift 2;;
    --num_graphs) NUM_GRAPHS="$2"; shift 2;;
    --m) M="$2"; shift 2;;
    --er_prob) ER="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;

    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --weight_decay) WD="$2"; shift 2;;

    --anneal_steps) ANNEAL_STEPS="$2"; shift 2;;
    --anneal_T0) ANNEAL_T0="$2"; shift 2;;
    --anneal_Tend) ANNEAL_TEND="$2"; shift 2;;

    --overwrite) OVERWRITE=1; shift 1;;

    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$ROUND" ]]; then
  echo "Missing --round"
  exit 1
fi

if [[ -z "${SPLIT_PATH}" ]]; then
  SPLIT_PATH="outputs/splits/split_seed${SEED}.json"
fi

if [[ "${TAU_HAT_SOURCE}" == "file" && -z "${TAU_HAT_PATH}" ]]; then
  echo "When --tau_hat_source file, you must provide --tau_hat_path"
  exit 1
fi

export PYTHONPATH="$(pwd)"

EXTRA_ARGS=()
if [[ "${OVERWRITE}" -eq 1 ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

CMD=(python -m src.pipeline.run_round
  --round "$ROUND"
  --tau_hat_source "$TAU_HAT_SOURCE"
  --n_nodes "$N_NODES"
  --num_graphs "$NUM_GRAPHS"
  --m "$M"
  --er_prob "$ER"
  --seed "$SEED"
  --epochs "$EPOCHS"
  --lr "$LR"
  --weight_decay "$WD"
  --split_path "$SPLIT_PATH"
  --anneal_steps "$ANNEAL_STEPS"
  --anneal_T0 "$ANNEAL_T0"
  --anneal_Tend "$ANNEAL_TEND"
)

if [[ "${TAU_HAT_SOURCE}" == "file" ]]; then
  CMD+=(--tau_hat_path "$TAU_HAT_PATH")
fi

CMD+=("${EXTRA_ARGS[@]}")

echo ">>> ${CMD[*]}"
"${CMD[@]}"

