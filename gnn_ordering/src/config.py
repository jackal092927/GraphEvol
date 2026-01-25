from dataclasses import dataclass

@dataclass(frozen=True)
class DataConfig:
    n_nodes: int = 30
    num_graphs: int = 120
    m: int = 2
    er_prob: float = 0.05
    seed: int = 7
    train_ratio: float = 0.70
    val_ratio: float = 0.15

@dataclass(frozen=True)
class PIConfig:
    bandwidth: float = 0.05
    resolution: tuple[int, int] = (12, 12)
    im_range: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)
    dims: tuple[int, ...] = (0, 1)

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.15
    patience: int = 30

@dataclass(frozen=True)
class SearchConfig:
    M: int = 200_000
    K: int = 10
    batch_size: int = 4096
    steps: int = 3000
    T0: float = 1.0
    Tend: float = 1e-3
