"""
apply_hybrid_fusion.py
Swaps specific `Concat` layers in an already-built Ultralytics model with
HybridBiAFFPNConcat (BiFPN-weighted fusion + AFFPN-style attention),
AFTER the model is constructed from a plain YAML that still has real
Concat layers (e.g. yolo26n_bifpn_affpn.yaml).

Why post-construction (same reasoning as apply_hwd.py):
Multi-input custom modules break Ultralytics' parse_model at YAML-parse
time (the "list indices must be integers or slices, not list" error you
already ran into). Doing the swap in Python after construction sidesteps
this: we run a real forward pass, capture the ACTUAL input channel counts
each Concat layer receives via a forward-pre-hook, then build
HybridBiAFFPNConcat with those exact numbers - zero guessing, zero
parse_model patching.

Usage in your notebook:

    from ultralytics import YOLO
    from hybrid_fusion import HybridBiAFFPNConcat
    from apply_hybrid_fusion import apply_hybrid_fusion

    model = YOLO("yolo26n_bifpn_affpn.yaml")

    # fusion points in yolo26n_bifpn_affpn.yaml are layer indices 12, 15, 18, 21
    apply_hybrid_fusion(model, layer_indices=[12, 15, 18, 21])

    # sanity check shapes before training
    import torch
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))
    print("OK:", [o.shape for o in out] if isinstance(out, (list, tuple)) else out.shape)

    # THEN load pretrained weights if you have them (the 4 fusion layers
    # you replaced won't transfer - expected, they're a new module)
    model.load("path/to/pretrained.pt")

    model.train(
        data="../../../YOLO.v1i.yolo26/data.yaml",
        epochs=500, imgsz=640, batch=16, optimizer="SGD",
        name="ablation_hybrid_bifpn_affpn",
    )
"""

import torch
from hybrid_fusion import HybridBiAFFPNConcat


def apply_hybrid_fusion(model_wrapper, layer_indices, c2_list=None, imgsz=640, verbose=True):
    """
    Args:
        model_wrapper: the YOLO(...) object (NOT model_wrapper.model - pass
            the full YOLO wrapper so we can run a dummy forward on it).
        layer_indices: list[int] of layer indices in model.model.model that
            are currently plain `Concat` layers (e.g. [12, 15, 18, 21] for
            yolo26n_bifpn_affpn.yaml).
        c2_list: optional list[int], one output-channel count per fusion
            point, same order as layer_indices. If omitted, defaults to
            sum(real input channels) for each point - this exactly matches
            what a normal Concat would have output, so every downstream
            layer (already built assuming that channel count) keeps working
            with zero further changes.
        imgsz: input size for the dummy forward pass used to detect real
            channel counts (must match the imgsz you intend to train with,
            or at least share the same channel-count structure - spatial
            size doesn't matter for channel counts, but keep it realistic).
        verbose: print a summary of what was replaced.
    """
    detection_model = model_wrapper.model
    layers = detection_model.model  # nn.Sequential

    for idx in layer_indices:
        cls_name = layers[idx].__class__.__name__
        assert cls_name == "Concat", (
            f"layer {idx} is a '{cls_name}', not 'Concat'. Check layer_indices "
            f"against your YAML's fusion-point comments."
        )

    captured = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inputs):
            xs = inputs[0]  # Concat.forward(self, x) receives x = list[Tensor]
            captured[idx] = [t.shape[1] for t in xs]
        return hook

    for idx in layer_indices:
        hooks.append(layers[idx].register_forward_pre_hook(make_hook(idx)))

    was_training = detection_model.training
    detection_model.eval()
    with torch.no_grad():
        detection_model(torch.zeros(1, 3, imgsz, imgsz))
    detection_model.train(was_training)

    for h in hooks:
        h.remove()

    for i, idx in enumerate(layer_indices):
        in_channels = captured[idx]
        old = layers[idx]
        c2 = c2_list[i] if c2_list else sum(in_channels)

        new = HybridBiAFFPNConcat(in_channels, c2)
        new.i = old.i
        new.f = old.f
        new.type = "hybrid_fusion.HybridBiAFFPNConcat"
        new.np = sum(x.numel() for x in new.parameters())

        layers[idx] = new
        if verbose:
            print(
                f"[apply_hybrid_fusion] layer {idx}: Concat({in_channels}->{sum(in_channels)}) "
                f"-> HybridBiAFFPNConcat({in_channels}->{c2})  params={new.np}"
            )

    return model_wrapper