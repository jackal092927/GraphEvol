from __future__ import annotations
from typing import Dict, List, Tuple
import os, json

SplitDict = Dict[str, List[str]]

def save_split(path: str, split: SplitDict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

def load_split(path: str) -> SplitDict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for k in ["train_graph_ids", "val_graph_ids", "test_graph_ids"]:
        if k not in obj:
            raise ValueError(f"split file missing key: {k}")
    return obj

def apply_split(instances, split: SplitDict):
    by_id = {inst.graph_id: inst for inst in instances}
    def pick(ids: List[str]):
        out = []
        for gid in ids:
            if gid not in by_id:
                raise KeyError(f"graph_id {gid} in split file not found in generated instances")
            out.append(by_id[gid])
        return out
    train = pick(split["train_graph_ids"])
    val = pick(split["val_graph_ids"])
    test = pick(split["test_graph_ids"])
    return train, val, test
