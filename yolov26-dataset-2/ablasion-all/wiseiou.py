"""
wise_iou.py
Wise-IoU (WIoU) bounding box regression loss for Ultralytics YOLO.

Reference: "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing
Mechanism" (Tong et al., 2023).

Motivation: your mAP@0.5:0.95 has been stuck around 65-67.5 across every
architectural ablation (BiFPN, SPD-Conv, DySnakeConv, DCNv3, ECA, ConvSAM,
P2 head, ...) - a ceiling that doesn't move much no matter what module you
add. That pattern is consistent with a LOSS-side bottleneck rather than an
architecture-side one: Ultralytics' default box loss is CIoU, which adds
an aspect-ratio penalty term that can fight against precise localization
for low-quality / ambiguous ground-truth boxes (common in crack/dent
datasets where box boundaries are inherently fuzzy). Wise-IoU replaces
that penalty with:

  - a distance-based penalty (R_WIoU) whose denominator is detached from
    the graph, so it doesn't produce gradients that slow convergence on
    already-good boxes (a known CIoU issue)
  - (v3 only) a non-monotonic focusing coefficient that DOWN-weights both
    very easy AND very hard (likely low-quality-label) boxes, focusing
    gradient on the "moderate difficulty" boxes that benefit most from
    training - this directly targets noisy/fuzzy box boundaries.

This is a pure loss-function change - no YAML edits, no new nn.Module in
the architecture, fully orthogonal to every module you've tested so far.
You can combine it with ANY of your existing YAMLs (e.g. run your current
best architecture with WIoU instead of CIoU and see if mAP@0.5:0.95 moves).

--------------------------------------------------------------------------
HOW THE PATCH WORKS (same monkey-patch philosophy as your other custom
modules): Ultralytics' BboxLoss.forward (in ultralytics/utils/loss.py) does:

    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

We don't touch BboxLoss at all. Instead we replace the `bbox_iou` name
INSIDE the `ultralytics.utils.loss` module's namespace with a function that
computes the real Wise-IoU loss internally and returns `1 - loss_wiou`, so
the existing `(1.0 - iou)` arithmetic downstream reconstructs our loss
value unchanged. No library files are edited, no parse_model involved -
completely independent of your architecture-side registration headaches.

--------------------------------------------------------------------------
USAGE:

    from ultralytics import YOLO
    from wise_iou import patch_bbox_iou_with_wiou, unpatch_bbox_iou

    # v3 (recommended - non-monotonic focusing, best in the paper's own
    # ablations): monotonous=False
    patch_bbox_iou_with_wiou(monotonous=False)

    model = YOLO("yolo26n_spdconv_bifpn.yaml")   # ANY of your existing yamls
    model.load("path/to/pretrained.pt")
    model.train(
        data="../../../YOLO.v1i.yolo26/data.yaml",
        epochs=500, imgsz=640, batch=16, optimizer="SGD",
        name="ablation_wiou_v3_on_spdconv_bifpn",
    )

    # if you want to go back to standard CIoU later in the SAME kernel
    # session (e.g. to run a comparison), call:
    unpatch_bbox_iou()
--------------------------------------------------------------------------

BUG FIX HISTORY:
An earlier version of this file had no bounds on the WIoU v3 focusing
coefficient `r`. Because `r` involves a ratio of two quantities that can
each approach 0 (a per-box loss over a running-mean loss), an unlucky
early batch could send `r` - and therefore the box loss - into the
hundreds/thousands. That doesn't just hurt localization: a loss spike
that large dominates the optimizer step and corrupts the SHARED
backbone/neck weights too, which is why a real run collapsed to roughly
HALF of baseline on every metric (Precision, Recall, mAP@0.5,
mAP@0.5:0.95 all together) rather than showing an isolated box-loss
problem. This version clamps beta, r, and the final loss value to keep
everything in a numerically sane range - functionally equivalent to the
paper's intent (still non-monotonic focusing) but without the blowup risk.
"""

import torch


class WiseIoULoss:
    """Computes the Wise-IoU loss value (NOT the IoU itself - see module
    docstring for why). Keeps a running EMA of the mean IoU-loss across
    batches, needed for the v3 non-monotonic focusing coefficient.

    Args:
        monotonous: False -> WIoU v3 (distance penalty + non-monotonic
            focusing coefficient, recommended). True -> WIoU v1 (distance
            penalty only, simpler/cheaper, no running-mean state).
        alpha, delta: WIoU v3 focusing-coefficient constants. Paper's
            reported best values (alpha=1.9, delta=3) are the defaults;
            rarely need changing.
        momentum: EMA momentum for the running mean of L_IoU (v3 only).
    """

    def __init__(self, monotonous=False, alpha=1.9, delta=3.0, momentum=1e-2,
                 beta_clip=8.0, r_clip=3.0, loss_clip=2.0):
        self.monotonous = monotonous
        self.alpha = alpha
        self.delta = delta
        self.momentum = momentum
        # --- stability clamps (see BUG FIX note below) ---
        self.beta_clip = beta_clip   # cap on the outlier-degree ratio
        self.r_clip = r_clip         # cap on the focusing coefficient itself
        self.loss_clip = loss_clip   # cap on the final per-box loss value
        # start iou_mean at a sane prior (1.0 = "assume average box is
        # currently a total miss") instead of letting the FIRST batch set
        # it - if that first batch happens to have a very small mean loss
        # (by chance, or because of a warm restart), beta = loss/iou_mean
        # for every subsequent batch explodes, which is what caused the
        # collapsed run you saw (Precision/Recall/mAP all roughly halved -
        # a training-wide loss-scale blowup, not a WIoU-specific box issue).
        self.iou_mean = 1.0

    def __call__(self, pred, target, xywh=False, eps=1e-7):
        if xywh:
            x1, y1, w1, h1 = pred.unbind(-1)
            x2, y2, w2, h2 = target.unbind(-1)
            b1_x1, b1_x2 = x1 - w1 / 2, x1 + w1 / 2
            b1_y1, b1_y2 = y1 - h1 / 2, y1 + h1 / 2
            b2_x1, b2_x2 = x2 - w2 / 2, x2 + w2 / 2
            b2_y1, b2_y2 = y2 - h2 / 2, y2 + h2 / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = pred.unbind(-1)
            b2_x1, b2_y1, b2_x2, b2_y2 = target.unbind(-1)

        # intersection / union / IoU
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        w1_, h1_ = (b1_x2 - b1_x1), (b1_y2 - b1_y1)
        w2_, h2_ = (b2_x2 - b2_x1), (b2_y2 - b2_y1)
        union = w1_ * h1_ + w2_ * h2_ - inter + eps
        iou = inter / union

        # enclosing box diagonal (denominator of the distance penalty -
        # detached, per the paper, so it doesn't itself produce gradients)
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = (cw ** 2 + ch ** 2) + eps

        # center-point distance
        cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
        rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

        L_iou = 1.0 - iou
        R_wiou = torch.exp(rho2 / c2.detach())
        L_wiou_v1 = R_wiou * L_iou

        if self.monotonous:
            # v1 is already bounded (IoU in [0,1], R_wiou <= e^1 since
            # rho <= c by construction) - clamp defensively anyway.
            return L_wiou_v1.clamp(min=0.0, max=self.loss_clip)

        # --- WIoU v3: non-monotonic focusing coefficient ---
        # BUG FIX: beta = L_iou / iou_mean is a ratio of two numbers that
        # can each independently approach 0, so beta (and therefore r) is
        # UNBOUNDED in principle. The original implementation had no cap,
        # so an unlucky early batch (small iou_mean) could make beta - and
        # r - spike into the hundreds/thousands, blowing up the box loss
        # and derailing the ENTIRE model (not just localization - this is
        # exactly the "everything roughly halved" collapse you saw, since
        # a huge loss on one term dominates the optimizer step and wrecks
        # shared backbone/neck weights too).
        L_iou_detached = L_iou.detach()
        batch_mean = L_iou_detached.mean().item()
        self.iou_mean = (1 - self.momentum) * self.iou_mean + self.momentum * batch_mean
        # keep iou_mean away from 0 - a near-zero denominator is the direct
        # cause of beta blowing up
        safe_mean = max(self.iou_mean, 0.1)

        beta = L_iou_detached / (safe_mean + eps)
        beta = beta.clamp(max=self.beta_clip)  # cap the outlier-degree ratio itself

        r = beta / (self.delta * self.alpha ** (beta - self.delta))
        r = r.clamp(max=self.r_clip)  # cap the focusing coefficient

        loss = r * L_wiou_v1
        return loss.clamp(min=0.0, max=self.loss_clip)


_ORIGINAL_BBOX_IOU = None  # stores the real bbox_iou so we can restore it


def patch_bbox_iou_with_wiou(monotonous=False, alpha=1.9, delta=3.0, momentum=1e-2):
    """Monkey-patches ultralytics.utils.loss.bbox_iou to compute Wise-IoU
    instead of the default CIoU, without touching any library files.

    Call this ONCE, before building/training your model. Returns the
    WiseIoULoss instance (rarely needed directly, but handy if you want to
    inspect wiou.iou_mean during training for debugging).
    """
    global _ORIGINAL_BBOX_IOU
    import ultralytics.utils.loss as loss_module

    if _ORIGINAL_BBOX_IOU is None:
        _ORIGINAL_BBOX_IOU = loss_module.bbox_iou

    wiou = WiseIoULoss(monotonous=monotonous, alpha=alpha, delta=delta, momentum=momentum)

    def patched_bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, **kwargs):
        loss = wiou(box1, box2, xywh=xywh, eps=eps)
        # loss is already clamped to [0, loss_clip] inside WiseIoULoss, so
        # (1 - loss) stays within a sane range for BboxLoss's (1.0 - iou)
        # arithmetic - no more runaway/negative values reaching the optimizer.
        return 1.0 - loss

    loss_module.bbox_iou = patched_bbox_iou
    print(
        f"[wise_iou] patched ultralytics.utils.loss.bbox_iou -> "
        f"Wise-IoU {'v1 (monotonous)' if monotonous else 'v3 (non-monotonic focusing)'}"
    )
    return wiou


def patch_bbox_iou_with_wiou_from_yaml(config_path="wise_iou_config.yaml"):
    """Same as patch_bbox_iou_with_wiou(), but reads hyperparameters from
    a YAML config file (see wise_iou_config.yaml) instead of hardcoding
    them in your notebook. Handy for keeping loss-function settings
    version-controlled/separate from your training script, and for
    quickly swapping between different WIoU configs across runs.

    Usage:
        from wise_iou import patch_bbox_iou_with_wiou_from_yaml
        patch_bbox_iou_with_wiou_from_yaml("wise_iou_config.yaml")
    """
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    return patch_bbox_iou_with_wiou(
        monotonous=cfg.get("monotonous", False),
        alpha=cfg.get("alpha", 1.9),
        delta=cfg.get("delta", 3.0),
        momentum=cfg.get("momentum", 1e-2),
    )


def unpatch_bbox_iou():
    """Restores the original ultralytics bbox_iou (standard CIoU/DIoU/GIoU
    behavior). Useful if you want to run a CIoU-baseline comparison in the
    same notebook session without restarting the kernel."""
    global _ORIGINAL_BBOX_IOU
    import ultralytics.utils.loss as loss_module

    if _ORIGINAL_BBOX_IOU is None:
        print("[wise_iou] nothing to restore - bbox_iou was never patched.")
        return
    loss_module.bbox_iou = _ORIGINAL_BBOX_IOU
    print("[wise_iou] restored original bbox_iou (standard CIoU).")