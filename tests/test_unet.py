"""Tests for the segmentation network.

The parameter-budget test is the important one: < 2M parameters is a project
constraint because the model has to be credible as an FPGA payload. A refactor
that quietly pushes it to 3M would invalidate the project's central claim, not
just slow things down, so it fails the build.
"""

from __future__ import annotations

import pytest
import torch

from iceroute.model.unet import UNet, UNetConfig, build_model

PARAM_BUDGET = 2_000_000


def test_parameter_budget():
    model = UNet()
    n = model.param_count()
    assert n < PARAM_BUDGET, (
        f"The model has {n:,} parameters, over the budget of {PARAM_BUDGET:,} "
        f"(docs/decisions.md D-005). Reduce encoder_channels before the depth."
    )


def test_forward_shape_preserved():
    """Segmentation output must be the same H×W as the input."""
    model = UNet(UNetConfig(in_channels=3, n_classes=6)).eval()
    x = torch.zeros(2, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 6, 256, 256)


@pytest.mark.parametrize("size", [64, 128, 256])
def test_forward_various_sizes(size):
    """Whole-scene inference tiles the image; tail tiles are not always 256."""
    model = UNet().eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 3, size, size))
    assert y.shape[-2:] == (size, size)


def test_odd_size_does_not_crash():
    """A tile whose side is not a multiple of 16 must still round-trip.

    Four pooling stages divide by 16. The Up block pads to the skip's size to
    absorb the mismatch; this test is what proves that path works.
    """
    model = UNet().eval()
    with torch.no_grad():
        y = model(torch.zeros(1, 3, 100, 140))
    assert y.shape[-2:] == (100, 140)


def test_no_batchnorm_anywhere():
    """GroupNorm only.

    BatchNorm with a batch of 8 gives noisy statistics, and its train/eval
    behaviour differs, which is a silent source of train/deploy mismatch.
    """
    offenders = [
        name
        for name, module in UNet().named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert not offenders, f"BatchNorm layers found: {offenders}"


def test_no_exotic_layers():
    """The FPGA target allows conv/relu/pool/upsample only."""
    allowed = (
        torch.nn.Conv2d,
        torch.nn.GroupNorm,
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
        torch.nn.Upsample,
        torch.nn.Sequential,
        torch.nn.ModuleList,
    )
    offenders = [
        f"{name}: {type(m).__name__}"
        for name, m in UNet().named_modules()
        if name and not isinstance(m, allowed) and type(m).__module__.startswith("torch.nn")
    ]
    assert not offenders, f"Layers not allowed for the FPGA target: {offenders}"


def test_extra_input_channel_stays_in_budget():
    """The optional distance_to_land ablation must not blow the budget."""
    model = UNet(UNetConfig(in_channels=4))
    assert model.param_count() < PARAM_BUDGET


def test_build_model_rejects_unknown_keys():
    """A typo in configs/train.yaml should fail loudly, not be ignored."""
    with pytest.raises(ValueError, match="Unrecognised keys"):
        build_model({"in_channels": 3, "base_channels": 32})


def test_gradients_flow():
    """Every parameter must receive a gradient — catches a detached skip."""
    model = UNet(UNetConfig(n_classes=3))
    out = model(torch.randn(1, 3, 64, 64))
    out.mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"Parameters without a gradient: {missing}"
