# gnn_ordering (modular pipeline)

This repo modularizes the single-file notebook pipeline into reusable modules:
- data generation & splits
- ordering providers (degree / file / model-search)
- TDA + graph statistics features (`F_hat`)
- MLP training
- ordering search (random search / simulated annealing)
- pipeline entrypoints + bash scripts

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# oracle case:
bash scripts/run_oracle.sh --seed 7 \
  --n_nodes 30 --num_graphs 120 --m 2 --er_prob 0.05 \
  --epochs 200 --lr 1e-3 --weight_decay 1e-4 \
  --anneal_steps 3000 --anneal_T0 1.0 --anneal_Tend 0.001

# round0: degree-based tau_hat
bash scripts/run.sh --round 0 --tau_hat_source degree

# round1: reuse tau_hat saved from previous round
bash scripts/run.sh --round 1 --tau_hat_source file \
  --tau_hat_path outputs/orderings/round0/tau_hat_hat_trainval.jsonl

# please add --overwrite if you want to re-run any round.
```


Outputs land in `outputs/`.

## Notes
