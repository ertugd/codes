# bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedConcat(nn.Module):
    """
    BiFPN-style weighted concat (SAFE VERSION)

    YAML:
        - [[-1, 6], 1, Concat, [1]]
    """

    def __init__(self, dimension=1, epsilon=1e-4):
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon

        # learnable weights
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"WeightedConcat input list/tuple olmalı, gelen tip: {type(x)}")

        if len(x) != 2:
            raise ValueError(f"WeightedConcat şu an 2 giriş bekliyor, gelen giriş sayısı: {len(x)}")

        x1, x2 = x

        # 🔥 SHAPE FIX (çok önemli!)
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='nearest')

        # 🔥 NO INPLACE (leaf error fix)
        w = torch.relu(self.w)

        # normalize weights
        weight = w / (torch.sum(w) + self.epsilon)

        # concat
        return torch.cat(
            [weight[0] * x1, weight[1] * x2],
            dim=self.d
        )


def patch_concat_to_bifpn():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Concat = WeightedConcat
    modules.Concat = WeightedConcat

    return WeightedConcat