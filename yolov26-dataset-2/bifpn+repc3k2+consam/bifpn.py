# bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedConcat(nn.Module):
    """
    BiFPN-style weighted concat.

    YAML içinde Concat olarak kalır:
      - [[-1, 6], 1, Concat, [1]]

    Python tarafında Concat -> WeightedConcat patch edilir.
    """

    def __init__(self, dimension=1, epsilon=1e-4):
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon

        # Leaf parameter. Buna kesinlikle inplace işlem yapmıyoruz.
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"WeightedConcat input list/tuple olmalı, gelen tip: {type(x)}")

        if len(x) != 2:
            raise ValueError(f"WeightedConcat şu an 2 giriş bekliyor, gelen giriş sayısı: {len(x)}")

        # ÖNEMLİ:
        # inplace=False olmalı.
        # self.w üzerinde relu_(), clamp_(), += gibi inplace işlem yapılmamalı.
        w = F.relu(self.w, inplace=False)
        weight = w / (w.sum() + self.epsilon)

        return torch.cat(
            [
                weight[0] * x[0],
                weight[1] * x[1],
            ],
            dim=self.d
        )


def patch_concat_to_bifpn():
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Concat = WeightedConcat
    modules.Concat = WeightedConcat

    return WeightedConcat