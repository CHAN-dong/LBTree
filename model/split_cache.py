# model/split_cache.py
from __future__ import annotations
from dataclasses import dataclass
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

    def set_root(self, gcn, mlp):
        self.root_gcn = copy.deepcopy(gcn.state_dict())
        self.root_mlp = copy.deepcopy(mlp.state_dict())
        self.last_gcn = copy.deepcopy(self.root_gcn)
        self.last_mlp = copy.deepcopy(self.root_mlp)

    def get_init(self, use_root: bool = False):
        if use_root and self.root_gcn is not None and self.root_mlp is not None:
            return self.root_gcn, self.root_mlp
        # default: use last
        if self.last_gcn is not None and self.last_mlp is not None:
            return self.last_gcn, self.last_mlp
        return None, None

    def update_last(self, gcn, mlp):
        self.last_gcn = copy.deepcopy(gcn.state_dict())
        self.last_mlp = copy.deepcopy(mlp.state_dict())