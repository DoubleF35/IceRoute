"""U-Net for sea ice stage-of-development segmentation.

The constraints are not stylistic — each one exists because the target is an
FPGA payload on a CubeSat:

* **< 2M parameters.** A project budget, not a suggestion. ``param_count()``
  is asserted in ``tests/test_unet.py``.
* **GroupNorm, not BatchNorm.** With a batch of 8 on 6 GB of VRAM, BatchNorm
  statistics are noisy; GroupNorm is also batch-independent at inference, which
  removes a whole class of train/deploy discrepancy, and it folds into INT8
  quantization more cleanly.
* **No exotic layers.** conv 3×3, ReLU, max pool, nearest upsample. No
  attention, no deformable convolutions, no transformer blocks. Every operator
  here has a well-trodden path to HLS.

Upsampling is nearest-neighbour (parameter-free) followed by a convolution,
rather than a transposed convolution: transposed conv produces checkerboard
artefacts on categorical targets and is more awkward to quantize.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

# GroupNorm needs num_channels % num_groups == 0. Every channel count in the
# default configuration is a multiple of 8.
DEFAULT_GROUPS = 8


def _groups_for(channels: int, preferred: int = DEFAULT_GROUPS) -> int:
    """Largest divisor of ``channels`` that is <= ``preferred``."""
    g = min(preferred, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ConvBlock(nn.Module):
    """conv3×3 → GroupNorm → ReLU, twice. The U-Net workhorse."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = DEFAULT_GROUPS):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False: the GroupNorm affine shift immediately after makes the
            # convolution bias redundant, and it is one less tensor to quantize.
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups_for(out_ch, groups), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_groups_for(out_ch, groups), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Nearest upsample, concatenate the skip, then a ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, groups: int = DEFAULT_GROUPS):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, groups)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Odd input sizes make the upsampled tensor a pixel short of the skip.
        # Patches are 256×256 so this should never fire in training, but whole
        # scenes are tiled at inference and the tail tile can be any size.
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.pad(
                x,
                [0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2]],
            )
        return self.conv(torch.cat([skip, x], dim=1))


@dataclass
class UNetConfig:
    """Shape of the network. Mirrors the ``model:`` block of ``configs/train.yaml``.

    ``encoder_channels`` defaults to ``[16, 32, 64, 128]`` rather than the
    ``[32, 64, 128, 256]`` of the original design, which was only a starting
    point. At base 32 the model comes out around 4.7M parameters, well over the
    2M budget. The design rule fixed the remedy in advance — reduce base
    channels before reducing depth — so the depth of 4 is preserved and the
    width is halved. See docs/decisions.md D-005.
    """

    in_channels: int = 3  # hv, hh, incidence_angle
    n_classes: int = 6
    encoder_channels: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    bottleneck_channels: int = 256
    groups: int = DEFAULT_GROUPS


class UNet(nn.Module):
    """Encoder–decoder with skip connections.

    Parameters
    ----------
    cfg
        Shape configuration. The default is the one that fits the 2M budget.
    """

    def __init__(self, cfg: UNetConfig | None = None, **overrides):
        super().__init__()
        if cfg is None:
            cfg = UNetConfig(**overrides)
        elif overrides:
            raise TypeError("Pass either a UNetConfig or individual parameters, not both.")
        self.cfg = cfg

        enc_ch = list(cfg.encoder_channels)
        if len(enc_ch) < 2:
            raise ValueError("At least two encoder levels are required.")

        # --- Encoder --------------------------------------------------------
        self.encoders = nn.ModuleList()
        prev = cfg.in_channels
        for ch in enc_ch:
            self.encoders.append(ConvBlock(prev, ch, cfg.groups))
            prev = ch
        self.pool = nn.MaxPool2d(2)

        # --- Bottleneck -----------------------------------------------------
        self.bottleneck = ConvBlock(prev, cfg.bottleneck_channels, cfg.groups)

        # --- Decoder, mirroring the encoder ---------------------------------
        self.decoders = nn.ModuleList()
        prev = cfg.bottleneck_channels
        for ch in reversed(enc_ch):
            self.decoders.append(Up(prev, ch, ch, cfg.groups))
            prev = ch

        # --- Head -----------------------------------------------------------
        # No normalisation or activation: raw logits, so CrossEntropyLoss can
        # apply log_softmax itself in a numerically stable way.
        self.head = nn.Conv2d(prev, cfg.n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C_in, H, W)`` → ``(B, n_classes, H, W)`` logits."""
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips), strict=True):
            x = dec(x, skip)

        return self.head(x)

    # -- introspection -------------------------------------------------------

    def param_count(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def size_mb(self, bytes_per_param: int = 4) -> float:
        """Approximate on-disk size. ``bytes_per_param=1`` for the INT8 estimate.

        This is an estimate for sanity-checking during design. The number that
        gets quoted is the **measured** file size from ``model/quantize.py``
        (docs/decisions.md D-014).
        """
        return self.param_count(trainable_only=False) * bytes_per_param / 1024**2

    def summary(self) -> str:
        n = self.param_count()
        budget = 2_000_000
        status = "within budget" if n < budget else "OVER BUDGET"
        return (
            f"UNet  in={self.cfg.in_channels}  classes={self.cfg.n_classes}  "
            f"encoder={self.cfg.encoder_channels}  bottleneck={self.cfg.bottleneck_channels}\n"
            f"  parameters: {n:,} / {budget:,} ({status})\n"
            f"  fp32 ~{self.size_mb():.2f} MB · INT8 estimated ~{self.size_mb(1):.2f} MB"
        )


def build_model(cfg_dict: dict | None = None) -> UNet:
    """Build a UNet from the ``model:`` section of a parsed YAML config."""
    cfg_dict = dict(cfg_dict or {})
    known = {f for f in UNetConfig.__dataclass_fields__}
    unknown = set(cfg_dict) - known
    if unknown:
        raise ValueError(
            f"Unrecognised keys in the model config: {sorted(unknown)}. Allowed: {sorted(known)}"
        )
    return UNet(UNetConfig(**cfg_dict))


if __name__ == "__main__":
    model = UNet()
    print(model.summary())
    with torch.no_grad():
        out = model(torch.zeros(1, model.cfg.in_channels, 256, 256))
    print(f"  forward 1×{model.cfg.in_channels}×256×256 → {tuple(out.shape)}")
