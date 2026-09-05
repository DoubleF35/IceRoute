"""Export the trained U-Net to ONNX. SPEC §7.5, first half.

ONNX is the handover format to the FPGA toolchain, so this step is not a
formality: if the exported graph differs from the trained model, every number
measured downstream describes a model that was never trained.

That is what ``check_parity`` exists to prevent. It runs both graphs on the same
input and fails the export unless the logits agree to a tight tolerance *and*
the argmax agrees almost everywhere. Both checks are needed: small logit drift
is harmless, but a handful of pixels flipping class at a decision boundary is
exactly what a segmentation mIoU is sensitive to.

Shape policy
------------
Batch is dynamic; height and width are **fixed** at the tile size. Fully dynamic
spatial dimensions would be convenient here and a liability later — HLS
synthesis wants static shapes, and the whole pipeline already runs on fixed
tiles. Committing to that now means the ONNX we measure is the ONNX that ports.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from iceroute.model.unet import build_model

log = logging.getLogger("export")

# Logit tolerance. fp32 ONNX should be near-exact; anything above this means a
# real graph difference, not accumulation noise.
LOGIT_ATOL = 1e-4
# Fraction of pixels allowed to change class. Ties at a decision boundary can
# legitimately flip on a 1e-7 difference, so demanding zero would be brittle.
MAX_ARGMAX_DISAGREEMENT = 1e-4

OPSET = 17


@dataclass
class ParityReport:
    max_abs_logit_diff: float
    argmax_disagreement: float
    n_pixels: int
    passed: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "max_abs_logit_diff": float(self.max_abs_logit_diff),
            "argmax_disagreement": float(self.argmax_disagreement),
            "n_pixels": self.n_pixels,
            "passed": self.passed,
            "note": self.note,
        }


def export_onnx(
    model: torch.nn.Module,
    path: Path,
    *,
    in_channels: int,
    tile_size: int = 256,
    opset: int = OPSET,
) -> Path:
    """Write the fp32 ONNX graph."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    dummy = torch.zeros(1, in_channels, tile_size, tile_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        # Only the batch axis moves: see the module docstring.
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    log.info("ONNX fp32 scritto in %s (%.2f MB)", path, path.stat().st_size / 1024**2)
    return path


def onnx_session(path: Path, providers: list[str] | None = None, *, threads: int | None = None):
    """An ONNX Runtime session pinned to CPU unless told otherwise.

    CPU is the honest default here: the CubeSat has no CUDA device, so a latency
    measured on the laptop's GPU would describe hardware the payload does not
    have.

    ``threads=1`` makes the result bit-reproducible. Multi-threaded convolution
    reduces partial sums in whatever order the threads finish, so the low bits
    of a logit depend on machine load — which is fine for throughput and not
    fine for a gate.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path), sess_options=opts, providers=providers or ["CPUExecutionProvider"]
    )


def check_parity(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    in_channels: int,
    tile_size: int = 256,
    n_samples: int = 4,
    seed: int = 0,
    atol: float = LOGIT_ATOL,
    max_disagreement: float = MAX_ARGMAX_DISAGREEMENT,
) -> ParityReport:
    """Compare the torch model and the exported graph on random input.

    Random normal input is deliberate: the channels are standard-scaled, so
    N(0,1) covers the operating range without needing the dataset to be present.
    A zero input would pass trivially through any graph that got the shapes
    right, which is the bug this is meant to catch.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, in_channels, tile_size, tile_size)).astype("float32")

    model = model.eval().cpu()
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()

    # Single-threaded: this is a correctness gate, and it must give the same
    # answer on a busy laptop as on an idle one. Observed once in practice —
    # the parity check failed while a training run saturated the CPU, then
    # passed three times in a row once the machine was quiet.
    sess = onnx_session(onnx_path, threads=1)
    got = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})[0]

    if ref.shape != got.shape:
        return ParityReport(
            float("inf"), 1.0, 0, False,
            f"Shape diverse: torch {ref.shape} vs onnx {got.shape}.",
        )

    max_diff = float(np.abs(ref - got).max())
    disagree = float((ref.argmax(1) != got.argmax(1)).mean())
    n_px = int(np.prod(ref.argmax(1).shape))

    passed = max_diff <= atol and disagree <= max_disagreement
    note = (
        "Export fedele al modello addestrato."
        if passed
        else (
            f"Export NON fedele: differenza massima sui logit {max_diff:.2e} "
            f"(tolleranza {atol:.0e}), {disagree:.4%} di pixel cambiano classe "
            f"(tolleranza {max_disagreement:.4%}). Non usare questo file."
        )
    )
    return ParityReport(max_diff, disagree, n_px, passed, note)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Esporta il modello in ONNX fp32 e verifica la parità.")
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tile-size", type=int, default=None)
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = yaml.safe_load(args.config.read_text())
    ckpt_path = args.checkpoint or Path(cfg["output"]["best_checkpoint"])
    out_path = args.out or Path(cfg["output"]["onnx_fp32"])
    tile = args.tile_size or int(cfg.get("eval", {}).get("tile_size", 256))

    if not ckpt_path.exists():
        log.error("Checkpoint assente: %s. Lancia prima 'make train'.", ckpt_path)
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    in_ch = int(ckpt["config"]["model"]["in_channels"])

    export_onnx(model, out_path, in_channels=in_ch, tile_size=tile)
    parity = check_parity(model, out_path, in_channels=in_ch, tile_size=tile)

    log.info(
        "parità: max |Δlogit| %.2e, pixel discordanti %.4f%% → %s",
        parity.max_abs_logit_diff, 100 * parity.argmax_disagreement,
        "OK" if parity.passed else "FALLITA",
    )
    if not parity.passed:
        log.error(parity.note)

    record = {
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "size_mb": round(out_path.stat().st_size / 1024**2, 4),
        "opset": OPSET,
        "tile_size": tile,
        "in_channels": in_ch,
        "n_parameters": model.param_count(),
        "parity": parity.as_dict(),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    existing.setdefault("export", {})["fp32"] = record
    args.metrics.write_text(json.dumps(existing, indent=2))

    return 0 if parity.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
