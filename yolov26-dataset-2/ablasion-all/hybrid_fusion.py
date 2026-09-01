"""
hybrid_fusion.py
Hybrid BiFPN + AF-FPN fusion block for Ultralytics YOLO PAFPN-style heads.

Combines two ideas that were previously tested in isolation:
  1) BiFPN "fast normalized fusion": each input branch gets a learnable,
     non-negative scalar weight; branches are combined as a weighted sum
     (not a channel-wise concat) - w_i = relu(w_i), normalized so they
     sum to 1, output = sum(w_i * proj_i(x_i)).
  2) AF-FPN-style attention refinement: after the weighted sum, apply
     lightweight channel attention (ECA-style, no dimensionality
     reduction) followed by spatial attention (CBAM-style avg+max pool
     -> 7x7 conv -> sigmoid) to sharpen the fused feature further.

--------------------------------------------------------------------------
USAGE - matches your existing patch_concat_to_bifpn() pattern EXACTLY.
The YAML is NOT changed - it keeps plain `Concat` at every fusion point,
same as always. Call patch_concat_to_hybrid() ONCE, BEFORE building the
model (before `YOLO("your.yaml")`), same place/style you already call
patch_concat_to_bifpn():

    import torch
    from ultralytics import YOLO
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    from hybrid_fusion import patch_concat_to_hybrid

    patch_concat_to_hybrid()

    model = YOLO("yolo26n_p2head_spdconv.yaml")   # ANY yaml, unchanged,
                                                    # no layer_indices needed
    ...
    model.train(...)

How this works: patch_concat_to_hybrid() replaces the `Concat` name in
Ultralytics' own module namespaces (ultralytics.nn.modules, ultralytics.nn.tasks)
with HybridBiAFFPNConcat. When parse_model reads "Concat" out of the YAML
and resolves it via those namespaces, it gets OUR class instead - every
single Concat in the architecture becomes a hybrid BiFPN+AFFPN fusion
block automatically, with ZERO YAML edits and no per-layer index list
(works for 4 fusion points, 6 with a P2 head, however many the YAML has).

HybridBiAFFPNConcat itself is built LAZILY: since parse_model constructs
it as `Concat(dimension)` (same signature as the real Concat - it doesn't
know per-layer channel counts at parse time), the actual per-branch
projection convs, fusion weights, and attention layers are created on the
very FIRST forward pass, using the real input tensor shapes at that point.
This is the same lazy-build pattern your own bifpn.py's Concat replacement
already uses, so it fits directly into your existing pipeline.

Practical note: because the submodules don't exist until the first forward
pass, it's a good idea to run one dummy forward immediately after building
the model (before model.train()) to force early building - cheap
insurance against any framework code that inspects/copies the model
before training actually starts (e.g. EMA setup):

    model.model.eval()
    with torch.no_grad():
        model.model(torch.zeros(1, 3, 640, 640))
--------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv + BN + SiLU (local copy, no hard dependency on ultralytics)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ECABlock(nn.Module):
    """Efficient Channel Attention (see eca.py for the full write-up)."""

    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SpatialAttentionBlock(nn.Module):
    """CBAM-style spatial attention: pool across the channel dim (avg + max),
    concat, 7x7 conv, sigmoid, rescale."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class HybridBiAFFPNConcat(nn.Module):
    """Drop-in replacement for Ultralytics' `Concat` module. Same
    constructor signature (`dimension`), so parse_model can build it
    directly from a plain `Concat` YAML entry with zero changes.

    Internally: BiFPN-weighted sum of the input branches (instead of
    channel-wise concatenation), followed by AFFPN-style channel +
    spatial attention refinement. Output channel count defaults to
    sum(input channel counts) - i.e. exactly what a plain Concat would
    have produced - so every downstream layer (built by parse_model
    assuming normal Concat behavior) keeps working unmodified.

    Built LAZILY on the first forward call - see module docstring.
    """

    def __init__(self, dimension=1, eps=1e-4, eca_k=3, sa_kernel=7):
        super().__init__()
        self.d = dimension  # kept for API compatibility with Concat, unused internally
        self.eps = eps
        self.eca_k = eca_k
        self.sa_kernel = sa_kernel
        self._built = False
        self.n = None

    def _build(self, in_channels_list, device, dtype):
        c2 = sum(in_channels_list)
        self.proj = nn.ModuleList([Conv(c, c2, 1) for c in in_channels_list])
        self.w = nn.Parameter(torch.ones(len(in_channels_list), dtype=torch.float32))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.eca = ECABlock(self.eca_k)
        self.sa = SpatialAttentionBlock(self.sa_kernel)
        self.to(device=device, dtype=dtype)
        self.n = len(in_channels_list)
        self._built = True

    def forward(self, xs):
        if not self._built:
            in_channels_list = [t.shape[1] for t in xs]
            self._build(in_channels_list, xs[0].device, xs[0].dtype)

        assert len(xs) == self.n, (
            f"HybridBiAFFPNConcat built for {self.n} inputs but got {len(xs)}."
        )
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)

        fused = 0
        for i, x in enumerate(xs):
            fused = fused + w[i] * self.proj[i](x)

        fused = self.act(self.bn(fused))   # BiFPN weighted-fusion result
        fused = self.eca(fused)             # AFFPN part: channel attention
        fused = self.sa(fused)              # AFFPN part: spatial attention
        return fused


def patch_concat_to_hybrid():
    """Call ONCE, BEFORE building the model (before `YOLO("your.yaml")`).
    Replaces `Concat` in Ultralytics' own namespaces with
    HybridBiAFFPNConcat, so every Concat in ANY yaml you load afterward
    becomes a hybrid BiFPN+AFFPN fusion block - no YAML edits, no layer
    index list, same usage pattern as your existing patch_concat_to_bifpn().
    """
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules

    tasks.Concat = HybridBiAFFPNConcat
    modules.Concat = HybridBiAFFPNConcat
    if hasattr(modules, "conv") and hasattr(modules.conv, "Concat"):
        modules.conv.Concat = HybridBiAFFPNConcat

    print("[hybrid_fusion] patched Concat -> HybridBiAFFPNConcat (BiFPN weighted-fusion + AFFPN attention)")