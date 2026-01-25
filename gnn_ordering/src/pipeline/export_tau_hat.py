from __future__ import annotations
import os, json, argparse, pickle
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances_pkl", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--method", type=str, default="tau_hat")
    args = ap.parse_args()

    with open(args.instances_pkl, "rb") as f:
        instances = pickle.load(f)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for inst in instances:
            tau = np.array(inst.tau_hat, dtype=np.float64)
            order = np.argsort(tau).astype(int).tolist()
            f.write(json.dumps({"graph_id": inst.graph_id, "order": order, "method": args.method}) + "\n")

    print(f"Wrote {len(instances)} lines to {args.out_path}")

if __name__ == "__main__":
    main()
