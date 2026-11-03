# model/split_cache.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy
import torch

@dataclass
class SplitModelCache:
    """
    Cache model weights across splits for warm-start.
    - root_state: the trained root weights
    - last_state: last used weights (can be parent weights)
    """
    root_gcn: Optional[Dict[str, Any]] = None
    root_mlp: Optional[Dict[str, Any]] = None
    last_gcn: Optional[Dict[str, Any]] = None
    last_mlp: Optional[Dict[str, Any]] = None
    frozen_encoder: Any = None
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    root_epochs: int = 0
    nonroot_epochs: int = 0
    prepruned_nodes: int = 0
    cached_embeddings: Any = None
    training_calls: int = 0
    direct_accepted: int = 0
    direct_fallback: int = 0
    sample_object_epochs: int = 0

    def set_root(self, gcn, mlp):
        self.root_gcn = copy.deepcopy(gcn.state_dict())
        self.root_mlp = copy.deepcopy(mlp.state_dict())
        self.last_gcn = self.root_gcn
        self.last_mlp = copy.deepcopy(self.root_mlp)
        self.frozen_encoder = gcn

    def get_init(self, use_root: bool = False):
        if use_root and self.root_gcn is not None and self.root_mlp is not None:
            return self.root_gcn, self.root_mlp
        # default: use last
        if self.last_gcn is not None and self.last_mlp is not None:
            return self.last_gcn, self.last_mlp
        return None, None

    def update_last(self, gcn, mlp):
        if gcn is self.frozen_encoder and all(not p.requires_grad for p in gcn.parameters()):
            self.last_gcn = self.root_gcn
        else:
            self.last_gcn = copy.deepcopy(gcn.state_dict())
        self.last_mlp = copy.deepcopy(mlp.state_dict())


class BestValidSplit:
    """Keep the actual pre-update weights of the best nonempty split seen."""
    def __init__(self):
        self.loss = float('inf')
        self.state = None

    def consider(self, model, loss, probabilities):
        score = float(loss.detach())
        labels = probabilities.detach() >= .5
        if score < self.loss and bool(labels.any()) and not bool(labels.all()):
            self.loss = score
            self.state = copy.deepcopy(model.state_dict())

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)
            return True
        return False
