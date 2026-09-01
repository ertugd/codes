"""
dsconv.py
Dynamic Snake Convolution (DSConv) for Ultralytics YOLO (YOLO26 / YOLOv8-style configs).

Paper: "Dynamic Snake Convolution based on Topological Geometric Constraints
for Tubular Structure Segmentation" (Qi et al., ICCV 2023).
Adapted here for crack / dent detection: cracks are thin, curved, discontinuous
structures - the same geometric profile as blood vessels / tubular structures,
which is exactly what DSConv was designed to follow.

This file defines two things:
    - DSConv        : the core dynamic-snake conv (single direction: x or y)
    - DySnakeConv    : the block you actually put in the YAML.
                        Combines a normal 3x3 conv branch + DSConv-x + DSConv-y,
                        then fuses them with a 1x1 conv back to c2 channels.
                        Drop-in replacement anywhere a Conv/C3k2 output is used,
                        commonly placed at the P3 (fine detail) branch.

YAML usage (args = [out_channels, kernel_size]):
    - [-1, 1, DySnakeConv, [256, 3]]

Notebook registration (do this in the SAME cell/place where you already
register SPDConv, before building the model from the YAML):

    import ultralytics.nn.tasks as tasks
    from dsconv import DySnakeConv
    tasks.DySnakeConv = DySnakeConv

    # also expose it in ultralytics.nn.modules namespace (some ultralytics
    # versions build their lookup table from there instead of tasks.py)
    import ultralytics.nn.modules as modules
    modules.DySnakeConv = DySnakeConv
    if hasattr(modules, "__all__"):
        modules.__all__ = tuple(modules.__all__) + ("DySnakeConv",)

    # make parse_model treat DySnakeConv like Conv/C3k2 for channel scaling
    # (width multiplier + make_divisible). This mirrors whatever you already
    # did for SPDConv - add "DySnakeConv" to the same set/tuple.
"""

import math
import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard Conv + BN + SiLU (local copy so this file has no hard
    dependency on ultralytics internals)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSC(object):
    """Builds the deformed coordinate map and does the bilinear-interpolated
    resample of the input feature map along the learned snake path."""

    def __init__(self, input_shape, kernel_size, extend_scope, morph):
        self.num_points = kernel_size
        self.width = input_shape[2]
        self.height = input_shape[3]
        self.morph = morph
        self.extend_scope = extend_scope
        self.num_batch = input_shape[0]
        self.num_channels = input_shape[1]

    def _coordinate_map_3D(self, offset, if_offset):
        device = offset.device
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        y_center = torch.arange(0, self.width).repeat([self.height])
        y_center = y_center.reshape(self.height, self.width)
        y_center = y_center.permute(1, 0)
        y_center = y_center.reshape([-1, self.width, self.height])
        y_center = y_center.repeat([self.num_points, 1, 1]).float().to(device)

        x_center = torch.arange(0, self.height).repeat([self.width])
        x_center = x_center.reshape(self.width, self.height)
        x_center = x_center.permute(0, 1)
        x_center = x_center.reshape([-1, self.width, self.height])
        x_center = x_center.repeat([self.num_points, 1, 1]).float().to(device)

        if self.morph == 0:
            y = torch.linspace(0, 0, 1)
            x = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            y, x = torch.meshgrid(y, x, indexing="ij")
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0).to(device)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0).to(device)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, 1, self.width, self.height]
            )
            y_new = y_new.reshape(
                [self.num_batch, self.num_points, self.width, self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, self.width, self.height]
            )

            if if_offset:
                y_offset = y_offset.permute(1, 0, 2, 3)
                y_offset_new = y_offset.detach().clone()
                center = int(self.num_points // 2)
                y_offset_new[center] = 0
                for index in range(1, center + 1):
                    y_offset_new[center + index] = (
                        y_offset_new[center + index - 1] + y_offset[center + index]
                    )
                    y_offset_new[center - index] = (
                        y_offset_new[center - index + 1] + y_offset[center - index]
                    )
                y_offset_new = y_offset_new.permute(1, 0, 2, 3).to(device)
                y_new = y_new.add(y_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height]
            )
            return y_new, x_new

        else:
            y = torch.linspace(
                -int(self.num_points // 2),
                int(self.num_points // 2),
                int(self.num_points),
            )
            x = torch.linspace(0, 0, 1)
            y, x = torch.meshgrid(y, x, indexing="ij")
            y_spread = y.reshape(-1, 1)
            x_spread = x.reshape(-1, 1)

            y_grid = y_spread.repeat([1, self.width * self.height])
            y_grid = y_grid.reshape([self.num_points, self.width, self.height])
            y_grid = y_grid.unsqueeze(0).to(device)

            x_grid = x_spread.repeat([1, self.width * self.height])
            x_grid = x_grid.reshape([self.num_points, self.width, self.height])
            x_grid = x_grid.unsqueeze(0).to(device)

            y_new = y_center + y_grid
            x_new = x_center + x_grid

            y_new = y_new.repeat(self.num_batch, 1, 1, 1)
            x_new = x_new.repeat(self.num_batch, 1, 1, 1)

            y_new = y_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, 1, self.num_points, self.width, self.height]
            )
            y_new = y_new.reshape(
                [self.num_batch, self.num_points, self.width, self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, self.num_points, self.width, self.height]
            )

            if if_offset:
                x_offset = x_offset.permute(1, 0, 2, 3)
                x_offset_new = x_offset.detach().clone()
                center = int(self.num_points // 2)
                x_offset_new[center] = 0
                for index in range(1, center + 1):
                    x_offset_new[center + index] = (
                        x_offset_new[center + index - 1] + x_offset[center + index]
                    )
                    x_offset_new[center - index] = (
                        x_offset_new[center - index + 1] + x_offset[center - index]
                    )
                x_offset_new = x_offset_new.permute(1, 0, 2, 3).to(device)
                x_new = x_new.add(x_offset_new.mul(self.extend_scope))

            y_new = y_new.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height]
            )
            x_new = x_new.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height]
            )
            return y_new, x_new

    def _bilinear_interpolate_3D(self, input_feature, y, x):
        device = input_feature.device
        y = y.reshape([-1]).float()
        x = x.reshape([-1]).float()

        zero = torch.zeros([]).int()
        max_y = self.width - 1
        max_x = self.height - 1

        y0 = torch.floor(y).int()
        y1 = y0 + 1
        x0 = torch.floor(x).int()
        x1 = x0 + 1

        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        input_feature_flat = input_feature.flatten()
        input_feature_flat = input_feature_flat.reshape(
            self.num_batch, self.num_channels, self.width, self.height
        )
        input_feature_flat = input_feature_flat.permute(0, 2, 3, 1)
        input_feature_flat = input_feature_flat.reshape(-1, self.num_channels)
        dimension = self.height * self.width

        base = torch.arange(self.num_batch) * dimension
        base = base.reshape([-1, 1]).float()

        repeat = (
            torch.ones([self.num_points * self.width * self.height]).unsqueeze(0)
            if self.morph == 0
            else torch.ones([self.width * self.num_points * self.height]).unsqueeze(0)
        )
        base = torch.matmul(base, repeat)
        base = base.reshape([-1])
        base = base.to(device)

        base_y0 = base + y0 * self.height
        base_y1 = base + y1 * self.height

        index_a0 = base_y0 - base_y0 + base_y0 + x0
        index_c0 = base_y0 - base_y0 + base_y0 + x1
        index_a1 = base_y1 - base_y1 + base_y1 + x0
        index_c1 = base_y1 - base_y1 + base_y1 + x1

        value_a0 = input_feature_flat[index_a0.type(torch.int64)].to(device)
        value_c0 = input_feature_flat[index_c0.type(torch.int64)].to(device)
        value_a1 = input_feature_flat[index_a1.type(torch.int64)].to(device)
        value_c1 = input_feature_flat[index_c1.type(torch.int64)].to(device)

        y0_float = y0.float()
        y1_float = y1.float()
        x0_float = x0.float()
        x1_float = x1.float()

        vol_a0 = ((y1_float - y) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c0 = ((y1_float - y) * (x - x0_float)).unsqueeze(-1).to(device)
        vol_a1 = ((y - y0_float) * (x1_float - x)).unsqueeze(-1).to(device)
        vol_c1 = ((y - y0_float) * (x - x0_float)).unsqueeze(-1).to(device)

        outputs = value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 + value_c1 * vol_c1

        if self.morph == 0:
            outputs = outputs.reshape(
                [self.num_batch, self.num_points * self.width, 1 * self.height, self.num_channels]
            )
            outputs = outputs.permute(0, 3, 1, 2)
        else:
            outputs = outputs.reshape(
                [self.num_batch, 1 * self.width, self.num_points * self.height, self.num_channels]
            )
            outputs = outputs.permute(0, 3, 1, 2)
        return outputs

    def deform_conv(self, input_feature, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        deformed_feature = self._bilinear_interpolate_3D(input_feature, y, x)
        return deformed_feature


class DSConvCore(nn.Module):
    """Single-direction dynamic snake conv (morph=0 -> along x/width axis,
    morph=1 -> along y/height axis)."""

    def __init__(self, in_ch, out_ch, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        if morph == 0:
            self.dsc_conv = nn.Conv2d(
                in_ch, out_ch, kernel_size=(kernel_size, 1), stride=(kernel_size, 1), padding=0
            )
        else:
            self.dsc_conv = nn.Conv2d(
                in_ch, out_ch, kernel_size=(1, kernel_size), stride=(1, kernel_size), padding=0
            )

        groups = out_ch // 4 if out_ch >= 4 else 1
        self.gn = nn.GroupNorm(groups, out_ch)
        self.act = nn.SiLU()
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, x):
        offset = self.offset_conv(x)
        offset = self.bn(offset)
        offset = torch.tanh(offset)
        dsc = DSC(x.shape, self.kernel_size, self.extend_scope, self.morph)
        deformed_feature = dsc.deform_conv(x, offset, self.if_offset)
        out = self.dsc_conv(deformed_feature)
        out = self.gn(out)
        out = self.act(out)
        return out


class DySnakeConv(nn.Module):
    """The block to put directly in the YAML.

    3 parallel branches on the input:
      - a normal kxk Conv         (keeps standard local texture)
      - DSConvCore along x-axis   (follows horizontal/curved crack segments)
      - DSConvCore along y-axis   (follows vertical/curved crack segments)
    Branches are concatenated on channel dim and fused back to c2 with a 1x1 Conv.

    Args in YAML: [c2, k]   e.g. [-1, 1, DySnakeConv, [256, 3]]
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.conv_0 = Conv(c1, c2, k)
        self.conv_x = DSConvCore(c1, c2, kernel_size=k, morph=0)
        self.conv_y = DSConvCore(c1, c2, kernel_size=k, morph=1)
        self.fuse = Conv(c2 * 3, c2, 1)

    def forward(self, x):
        x0 = self.conv_0(x)
        xx = self.conv_x(x)
        xy = self.conv_y(x)
        out = torch.cat([x0, xx, xy], dim=1)
        return self.fuse(out)