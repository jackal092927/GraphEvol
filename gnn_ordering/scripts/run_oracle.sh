#!/usr/bin/env bash
set -euo pipefail

# Example:
#   bash scripts/run_oracle.sh --seed 7

SEED=7
N_NODES=30
NUM_GRAPHS=120
M=2
ER=0.05

EPOCHS=200
LR=1e-3
WD=1e-4
PATIENCE=30
HIDDEN1=256
HIDDEN2=128
DROPOUT=0.15

ANNEAL_STEPS=3000
ANNEAL_T0=1.0
ANNEAL_TEND=0.001

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --n_nodes) N_NODES="$2"; shift 2;;
    --num_graphs) NUM_GRAPHS="$2"; shift 2;;
    --m) M="$2"; shift 2;;
    --er_prob) ER="$2"; shift 2;;

    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --weight_decay) WD="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --hidden1) HIDDEN1="$2"; shift 2;;
    --hidden2) HIDDEN2="$2"; shift 2;;
    --dropout) DROPOUT="$2"; shift 2;;

    --anneal_steps) ANNEAL_STEPS="$2"; shift 2;;
    --anneal_T0) ANNEAL_T0="$2"; shift 2;;
    --anneal_Tend) ANNEAL_TEND="$2"; shift 2;;

    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

export PYTHONPATH="$(pwd)"

SPLIT_PATH="outputs/oracle_splits/split_seed${SEED}.json"
DATA_DIR="outputs/oracle_datasets/seed${SEED}"
CKPT_DIR="outputs/oracle_checkpoints/seed${SEED}"
ORD_DIR="outputs/oracle_orderings/seed${SEED}"
EVAL_DIR="outputs/oracle_evals/seed${SEED}"

mkdir -p "$(dirname "$SPLIT_PATH")" "$DATA_DIR" "$CKPT_DIR" "$ORD_DIR" "$EVAL_DIR"

echo "=== ORACLE RUN ==="
echo "seed=$SEED  n_nodes=$N_NODES  num_graphs=$NUM_GRAPHS  m=$M  er_prob=$ER"
echo "split_path=$SPLIT_PATH"
echo "data_dir=$DATA_DIR"
echo "ckpt_dir=$CKPT_DIR"
echo "ord_dir=$ORD_DIR"
echo "eval_dir=$EVAL_DIR"
echo "=================="

echo
echo ">>> [1] make_dataset_oracle"
python -m src.pipeline.make_dataset_oracle \
  --out_dir "$DATA_DIR" \
  --n_nodes "$N_NODES" --num_graphs "$NUM_GRAPHS" --m "$M" --er_prob "$ER" \
  --seed "$SEED" \
  --split_path "$SPLIT_PATH"

# -------------------------------------------------------------------------
# Compatibility bridge for existing run_train.py:
# run_train.py loads: data_dir/train.npz and data_dir/val.npz
# oracle dataset writes: oracle_train.npz / oracle_val.npz / oracle_test.npz
# So we symlink oracle_* to train/val/test.
# -------------------------------------------------------------------------

ln -sf oracle_train.npz "$DATA_DIR/train.npz"
ln -sf oracle_val.npz   "$DATA_DIR/val.npz"
ln -sf oracle_test.npz  "$DATA_DIR/test.npz"
echo
echo ">>> [2] train (run_train expects train.npz/val.npz; symlinks are set)"
python -m src.pipeline.run_train \
  --data_dir "$DATA_DIR" \
  --out_dir "$CKPT_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
  --patience "$PATIENCE" --hidden1 "$HIDDEN1" --hidden2 "$HIDDEN2" --dropout "$DROPOUT"

CKPT_PATH="$CKPT_DIR/model.pt"
INST_PKL="$DATA_DIR/instances.pkl"

echo
echo ">>> [3] anneal (trainval) with consistent F(order)"
python -m src.pipeline.run_anneal_batch_oracle \
  --ckpt "$CKPT_PATH" \
  --instances_pkl "$INST_PKL" \
  --split_path "$SPLIT_PATH" \
  --which trainval \
  --out_jsonl "$ORD_DIR/oracle_tau_hat_hat_trainval.jsonl" \
  --out_eval "$EVAL_DIR/oracle_trainval_eval.json" \
  --seed "$SEED" --steps "$ANNEAL_STEPS" --T0 "$ANNEAL_T0" --Tend "$ANNEAL_TEND"

echo
echo ">>> [4] anneal (test) with consistent F(order)"
python -m src.pipeline.run_anneal_batch_oracle \
  --ckpt "$CKPT_PATH" \
  --instances_pkl "$INST_PKL" \
  --split_path "$SPLIT_PATH" \
  --which test \
  --out_jsonl "$ORD_DIR/oracle_tau_hat_hat_test.jsonl" \
  --out_eval "$EVAL_DIR/oracle_test_eval.json" \
  --seed "$SEED" --steps "$ANNEAL_STEPS" --T0 "$ANNEAL_T0" --Tend "$ANNEAL_TEND"

echo
echo ">>> [5] root identify (test) with consistent F(order)"
python -m src.pipeline.run_root_identify_oracle \
  --ckpt "$CKPT_PATH" \
  --instances_pkl "$INST_PKL" \
  --split_path "$SPLIT_PATH" \
  --out_jsonl "$ORD_DIR/oracle_root_identify_test.jsonl" \
  --out_eval "$EVAL_DIR/oracle_root_identify_test_eval.json" \
  --seed "$SEED" --steps "$ANNEAL_STEPS" --T0 "$ANNEAL_T0" --Tend "$ANNEAL_TEND"

echo
echo "=== ORACLE DONE ==="
echo "ckpt:          $CKPT_PATH"
echo "trainval ord:  $ORD_DIR/oracle_tau_hat_hat_trainval.jsonl"
echo "test ord:      $ORD_DIR/oracle_tau_hat_hat_test.jsonl"
echo "test root:     $ORD_DIR/oracle_root_identify_test.jsonl"
echo "===================="

