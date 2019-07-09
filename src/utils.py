from typing import Dict

import torch
from torch import Tensor
from numpy import ndarray


def format_observed_state(state: Dict[str, ndarray]) -> Dict[str, Tensor]:
    for key, val in state.items():
        state[key] = torch.FloatTensor(val)

    return state
