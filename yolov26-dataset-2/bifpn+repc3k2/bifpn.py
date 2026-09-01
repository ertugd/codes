# bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedConcat(nn.Module):
    """
    BiFPN-style weighted concat.

    YAML içinde normal Concat olarak kullanılır:
      - [[-1, 7], 1, Concat, [1]]

    Python tarafında Concat sınıfını bununla değiştiriyoruz.
    """

    def __init__(self, dimension=1, epsilon=1e-4):
        super().__init__()

        self.d = dimension
        self.epsilon = epsilon

        # 2 girişli concat için öğrenilebilir ağırlık
        # Leaf variable üzerinde inplace işlem yapılmamalı.
        self.w = nn.Parameter(
            torch.ones(2, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(
                f"WeightedConcat input list/tuple olmalı, gelen tip: {type(x)}"
            )

        if len(x) != 2:
            raise ValueError(
                f"WeightedConcat şu an 2 giriş bekliyor, gelen giriş sayısı: {len(x)}"
            )

        # Inplace işlem yok.
        w = F.relu(self.w, inplace=False)
        weight = w / (w.sum() + self.epsilon)

        return torch.cat(
            [
                weight[0] * x[0],
                weight[1] * x[1]
            ],
            dim=self.d
        )


def patch_concat_to_bifpn():
    """
    Ultralytics Concat katmanını WeightedConcat ile değiştirir.

    Bu fonksiyon YOLO("...yaml") satırından önce çağrılmalı.
    """

    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Concat = WeightedConcat
    modules.Concat = WeightedConcat

    # Bazı Ultralytics sürümlerinde Concat conv modülü altında da bulunabilir.
    try:
        import ultralytics.nn.modules.conv as conv_modules
        conv_modules.Concat = WeightedConcat
    except Exception:
        pass

    return WeightedConcat