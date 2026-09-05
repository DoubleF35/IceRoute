"""Post-training INT8 quantization and the measurements around it. SPEC §7.5.

This file exists to replace estimates with measurements. Every number it emits
is read off a file on disk or a clock, never computed from a parameter count.
That matters here more than anywhere else in the project: the pitch deck claims
``58 KB INT8 weights``, which implies about 59,000 parameters, and the model has
1,964,182. See DECISIONS.md A-02 — the number this script prints is the one that
goes on the slide.

What is measured, and how
-------------------------
* **Size** — ``stat()`` on both files. A ratio, not a guess.
* **Latency** — median over repeated runs after a warm-up, on CPU. Reported at
  ``intra_op_num_threads=1`` as well as with the default thread pool, because
  the laptop has 22 cores and the payload does not; quoting the 22-core figure
  for an FPGA target would be meaningless.
* **Memory** — peak RSS of the process sampled during inference, minus the
  baseline before the session is created. This includes the runtime's allocator
  arena, which is what actually has to fit in the payload's RAM.
* **Accuracy** — the INT8 graph is scored through *exactly* the same tiling and
  blending as fp32 (see ``inference.OnnxSessionModel``). Anything else would
  confound quantization error with a change of procedure.

Calibration uses real patches drawn from the **training** split. Calibrating on
validation or test data would leak the evaluation set into the model and make
the reported degradation optimistic.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from iceroute.data.loader import load_config, load_scene
from iceroute.data.patches import CHANNELS, PatchDataset
from iceroute.model.export import onnx_session
from iceroute.model.inference import OnnxSessionModel, predict_scene
from iceroute.model.metrics import IGNORE_INDEX, ConfusionMatrix

log = logging.getLogger("quantize")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class PatchCalibrationReader:
    """Feeds real patches to the static quantizer.

    Implements onnxruntime's ``CalibrationDataReader`` protocol without
    subclassing it, so importing this module does not require the quantization
    extras to be installed.

    Patches are drawn evenly across the calibration pool rather than taken from
    the front: the cache is ordered by scene, and the first 200 entries would
    all come from one or two scenes — likely one region, one season, one ice
    regime. The activation ranges derived from that would not represent the
    data the model actually sees.
    """

    def __init__(self, dataset: PatchDataset, n_samples: int, input_name: str):
        self.input_name = input_name
        total = len(dataset)
        n = min(n_samples, total)
        if n == 0:
            raise ValueError("Nessuna patch disponibile per la calibrazione.")
        idx = np.linspace(0, total - 1, n).astype(int)
        self._batches = [
            {input_name: np.asarray(dataset[int(i)][0], dtype="float32")[None]} for i in idx
        ]
        self.n_samples = n
        self._it = iter(self._batches)

    def get_next(self):
        return next(self._it, None)

    def rewind(self) -> None:
        self._it = iter(self._batches)


def quantize_model(
    fp32_path: Path,
    int8_path: Path,
    reader: PatchCalibrationReader,
    *,
    per_channel: bool = True,
    method: str = "minmax",
) -> Path:
    """Static PTQ to INT8 in QDQ format.

    QDQ rather than QOperator: it keeps explicit Quantize/DeQuantize nodes in
    the graph, which is both what the FPGA toolchains consume and what makes
    the quantization inspectable afterwards.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    if method not in methods:
        raise ValueError(f"Metodo di calibrazione sconosciuto: {method}. Usa {sorted(methods)}.")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=methods[method],
    )
    log.info(
        "INT8 scritto in %s (%.3f MB, calibrato su %d patch)",
        int8_path, int8_path.stat().st_size / 1024**2, reader.n_samples,
    )
    return int8_path


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class _RSSSampler:
    """Samples process RSS in a thread while something runs."""

    def __init__(self, interval: float = 0.01):
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = 0
        self.baseline_bytes = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._proc.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> _RSSSampler:
        self.baseline_bytes = self._proc.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def delta_mb(self) -> float:
        return (self.peak_bytes - self.baseline_bytes) / 1024**2

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1024**2


@dataclass
class LatencyResult:
    """Per-tile inference latency under one thread configuration."""

    threads: int | None
    median_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    runs: int
    batch_size: int
    peak_rss_mb: float
    rss_delta_mb: float

    def as_dict(self) -> dict:
        return {
            "threads": self.threads or "default",
            "median_ms_per_tile": round(self.median_ms, 3),
            "mean_ms_per_tile": round(self.mean_ms, 3),
            "stdev_ms": round(self.stdev_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "runs": self.runs,
            "batch_size": self.batch_size,
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "rss_delta_mb": round(self.rss_delta_mb, 1),
        }


def measure_latency(
    onnx_path: Path,
    *,
    in_channels: int,
    tile_size: int = 256,
    runs: int = 10,
    warmup: int = 3,
    batch_size: int = 1,
    threads: int | None = None,
) -> LatencyResult:
    """Median per-tile latency on CPU.

    Median, not mean: on a laptop a single scheduling hiccup skews a mean of ten
    runs badly, and the median is what a repeat of this measurement will
    reproduce. The mean and standard deviation are reported alongside so the
    spread is visible rather than hidden.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1

    sess = ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    name = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    rng = np.random.default_rng(0)
    x = rng.standard_normal((batch_size, in_channels, tile_size, tile_size)).astype("float32")

    for _ in range(warmup):
        sess.run([out], {name: x})

    times = []
    with _RSSSampler() as sampler:
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run([out], {name: x})
            times.append((time.perf_counter() - t0) * 1000.0 / batch_size)

    return LatencyResult(
        threads=threads,
        median_ms=statistics.median(times),
        mean_ms=statistics.fmean(times),
        stdev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_ms=min(times),
        runs=runs,
        batch_size=batch_size,
        peak_rss_mb=sampler.peak_mb,
        rss_delta_mb=sampler.delta_mb,
    )


@dataclass
class AccuracyComparison:
    """fp32 vs INT8 on identical data through an identical pipeline."""

    fp32_miou: float
    int8_miou: float
    fp32_binary_accuracy: float
    int8_binary_accuracy: float
    miou_drop: float
    max_acceptable_drop: float
    acceptable: bool
    n_scenes: int
    per_scene: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fp32_miou": round(self.fp32_miou, 4),
            "int8_miou": round(self.int8_miou, 4),
            "miou_drop": round(self.miou_drop, 4),
            "fp32_binary_accuracy": round(self.fp32_binary_accuracy, 4),
            "int8_binary_accuracy": round(self.int8_binary_accuracy, 4),
            "max_acceptable_drop": self.max_acceptable_drop,
            "acceptable": self.acceptable,
            "n_scenes": self.n_scenes,
            "per_scene": self.per_scene,
        }


def compare_accuracy(
    fp32_onnx: Path,
    int8_onnx: Path,
    scene_paths: list[Path],
    data_cfg: dict,
    *,
    n_classes: int,
    channels: list[str],
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 8,
    max_acceptable_drop: float = 0.03,
) -> AccuracyComparison:
    """Score both graphs on whole scenes and report the degradation."""
    fp32 = OnnxSessionModel(onnx_session(fp32_onnx))
    int8 = OnnxSessionModel(onnx_session(int8_onnx))

    cm32, cm8 = ConfusionMatrix(n_classes), ConfusionMatrix(n_classes)
    per_scene = []

    for path in scene_paths:
        scene = load_scene(path, data_cfg)
        x = scene.channels(channels)
        truth = scene.sod.astype("int64").copy()
        truth[~scene.mask_valid] = IGNORE_INDEX

        one32, one8 = ConfusionMatrix(n_classes), ConfusionMatrix(n_classes)
        for model, agg, single in ((fp32, cm32, one32), (int8, cm8, one8)):
            labels, _, _ = predict_scene(
                model, x, n_classes=n_classes, tile_size=tile_size, overlap=overlap,
                batch_size=batch_size, device="cpu", amp=False,
            )
            agg.update(truth, labels)
            single.update(truth, labels)

        per_scene.append({
            "scene_id": scene.meta.scene_id,
            "fp32_miou": round(one32.miou(), 4),
            "int8_miou": round(one8.miou(), 4),
            "drop": round(one32.miou() - one8.miou(), 4),
        })
        log.info(
            "  %s: fp32 %.4f → int8 %.4f (Δ %+.4f)",
            scene.meta.scene_id, one32.miou(), one8.miou(), one8.miou() - one32.miou(),
        )

    drop = cm32.miou() - cm8.miou()
    return AccuracyComparison(
        fp32_miou=cm32.miou(),
        int8_miou=cm8.miou(),
        fp32_binary_accuracy=cm32.binary_accuracy(),
        int8_binary_accuracy=cm8.binary_accuracy(),
        miou_drop=drop,
        max_acceptable_drop=max_acceptable_drop,
        acceptable=bool(np.isfinite(drop) and drop <= max_acceptable_drop),
        n_scenes=len(scene_paths),
        per_scene=per_scene,
    )


def size_report(fp32_path: Path, int8_path: Path, n_parameters: int) -> dict:
    """Measured file sizes and the ratio between them.

    ``bytes_per_parameter`` is included because it is the sanity check that
    catches a wrong claim instantly: INT8 weights cannot be far below 1 byte per
    parameter, so any headline size implying less is arithmetically impossible.
    """
    fp32_b, int8_b = fp32_path.stat().st_size, int8_path.stat().st_size
    return {
        "fp32_bytes": fp32_b,
        "fp32_mb": round(fp32_b / 1024**2, 4),
        "int8_bytes": int8_b,
        "int8_mb": round(int8_b / 1024**2, 4),
        "int8_kb": round(int8_b / 1024, 1),
        "compression_ratio": round(fp32_b / int8_b, 3) if int8_b else None,
        "n_parameters": n_parameters,
        "bytes_per_parameter_int8": round(int8_b / n_parameters, 4) if n_parameters else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quantizza in INT8 e misura size, latenza, mIoU.")
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--fp32", type=Path, default=None)
    ap.add_argument("--int8", type=Path, default=None)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/train"))
    ap.add_argument("--scenes", type=int, default=2, help="scene intere per il confronto mIoU")
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/metrics.json"))
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = load_config(args.data_config)
    qcfg = cfg.get("quantize", {})
    eval_cfg = cfg.get("eval", {})

    fp32_path = args.fp32 or Path(cfg["output"]["onnx_fp32"])
    int8_path = args.int8 or Path(cfg["output"]["onnx_int8"])
    if not fp32_path.exists():
        log.error(
            "ONNX fp32 assente: %s. Lancialo con 'make quantize', che esporta e poi "
            "quantizza: se la parità dell'export fallisce, make si ferma lì.",
            fp32_path,
        )
        return 1

    n_classes = int(cfg["model"]["n_classes"])
    in_ch = int(cfg["model"]["in_channels"])
    tile = int(eval_cfg.get("tile_size", 256))
    channels = list(cfg["data"].get("channels") or CHANNELS[:3])

    split_file = Path(data_cfg.get("split_file", "configs/split.yaml"))
    split = yaml.safe_load(split_file.read_text())["split"]

    # -- calibrate and quantize --------------------------------------------
    log.info("Calibrazione su patch di training…")
    ds = PatchDataset(cfg["data"]["cache_dir"], split["train"], channels=channels, augment=False)
    sess = onnx_session(fp32_path)
    reader = PatchCalibrationReader(
        ds, int(qcfg.get("calibration_patches", 200)), sess.get_inputs()[0].name
    )
    del sess

    quantize_model(
        fp32_path, int8_path, reader,
        per_channel=bool(qcfg.get("per_channel", True)),
        method=str(qcfg.get("method", "minmax")),
    )

    ckpt = torch.load(cfg["output"]["best_checkpoint"], map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in ckpt["model"].values())

    report: dict = {
        "calibration_patches": reader.n_samples,
        "method": str(qcfg.get("method", "minmax")),
        "per_channel": bool(qcfg.get("per_channel", True)),
        "size": size_report(fp32_path, int8_path, n_params),
    }
    s = report["size"]
    log.info(
        "dimensioni: fp32 %.3f MB → int8 %.3f MB (%.1f KB), rapporto %.2f×, "
        "%.3f byte/parametro",
        s["fp32_mb"], s["int8_mb"], s["int8_kb"], s["compression_ratio"],
        s["bytes_per_parameter_int8"],
    )

    # -- latency ------------------------------------------------------------
    runs = int(qcfg.get("latency_runs", 10))
    report["latency"] = {}
    for label, threads in (("single_thread", 1), ("all_threads", None)):
        report["latency"][label] = {
            "fp32": measure_latency(
                fp32_path, in_channels=in_ch, tile_size=tile, runs=runs, threads=threads
            ).as_dict(),
            "int8": measure_latency(
                int8_path, in_channels=in_ch, tile_size=tile, runs=runs, threads=threads
            ).as_dict(),
        }
        f = report["latency"][label]["fp32"]["median_ms_per_tile"]
        q = report["latency"][label]["int8"]["median_ms_per_tile"]
        log.info(
            "latenza/tile (%s): fp32 %.1f ms → int8 %.1f ms (%.2f×)",
            label, f, q, f / q if q else float("nan"),
        )

    # -- accuracy -----------------------------------------------------------
    # No test split exists (SPEC §5.3); val is the non-training data we have.
    # For this particular comparison the selection bias does not matter: both
    # graphs are scored on identical pixels, so the *difference* is unaffected
    # even though each absolute mIoU is optimistic.
    scene_ids = [s for s in split["val"] if (args.raw_dir / s).exists()][: args.scenes]
    if args.skip_accuracy or not scene_ids:
        if not scene_ids and not args.skip_accuracy:
            log.warning("Nessuna scena di test scaricata: salto il confronto di accuratezza.")
        report["accuracy"] = {"skipped": True}
    else:
        log.info("Confronto fp32 vs INT8 su %d scene intere…", len(scene_ids))
        acc = compare_accuracy(
            fp32_path, int8_path, [args.raw_dir / s for s in scene_ids], data_cfg,
            n_classes=n_classes, channels=channels, tile_size=tile,
            overlap=int(eval_cfg.get("tile_overlap", 32)),
            batch_size=int(eval_cfg.get("batch_size", 8)),
            max_acceptable_drop=float(qcfg.get("max_acceptable_miou_drop", 0.03)),
        )
        report["accuracy"] = acc.as_dict()
        log.info(
            "mIoU: fp32 %.4f → int8 %.4f, degrado %+.4f (soglia %.3f) → %s",
            acc.fp32_miou, acc.int8_miou, -acc.miou_drop, acc.max_acceptable_drop,
            "accettabile" if acc.acceptable else "OLTRE SOGLIA",
        )
        if not acc.acceptable:
            log.warning(
                "Degrado oltre la soglia SPEC §7.5. L'opzione prevista è QAT con "
                "Brevitas, fuori percorso critico: prima di prenderla, verificare "
                "se il degrado si concentra su una classe sola."
            )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.metrics.read_text()) if args.metrics.exists() else {}
    existing["quantization"] = report
    args.metrics.write_text(json.dumps(existing, indent=2))
    log.info("Misure in %s", args.metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
