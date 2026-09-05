"""Train the U-Net. SPEC §7.2.

Constraints that shaped this file, all from the 6 GB of VRAM on the RTX 4050:

* AMP is mandatory, not optional. Without it the batch of 16 at 256x256 will
  not fit.
* On OOM the loop halves the batch size and retries rather than dying, and
  says so in the log. Losing a night's training to an OOM at epoch 40 is a
  worse outcome than a slower epoch.
* Logging goes to a CSV file. No cloud dependency: the demo laptop has to work
  with the network card switched off, and a training run that phones home is
  one more thing that can fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from iceroute.data.loader import load_config
from iceroute.data.patches import PatchDataset
from iceroute.model.losses import build_loss
from iceroute.model.metrics import ConfusionMatrix
from iceroute.model.unet import build_model

log = logging.getLogger("train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN would cost ~20% throughput for a reproducibility
    # guarantee we do not need; the seed is logged, which is what SPEC §7.2 asks.
    torch.backends.cudnn.benchmark = True


def cosine_with_warmup(step: int, *, warmup: int, total: int, min_factor: float = 0.01) -> float:
    """Linear warmup then cosine decay, as a multiplier on the base lr."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / (total - warmup)
    return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_ce: float
    train_dice: float
    val_loss: float
    val_miou: float
    val_accuracy: float
    val_binary_accuracy: float
    lr: float
    seconds: float
    peak_vram_gb: float


class CSVLogger:
    """Append-only CSV. Survives a crash; opens in any spreadsheet."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = path.exists() and path.stat().st_size > 0

    def append(self, row: EpochResult) -> None:
        d = row.__dict__
        with self.path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(d))
            if not self._header_written:
                w.writeheader()
                self._header_written = True
            w.writerow(d)


def collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch]))
    y = torch.from_numpy(np.stack([b[1] for b in batch]))
    m = torch.from_numpy(np.stack([b[2] for b in batch]))
    return x, y, m


def build_loaders(data_cfg: dict, train_cfg: dict, split: dict) -> tuple[DataLoader, DataLoader]:
    cache = Path(train_cfg["data"]["cache_dir"])
    channels = train_cfg.get("channels") or ["hh", "hv", "incidence_angle"]
    # The model config fixes how many channels the first conv expects; the two
    # must agree or the first forward pass fails with a shape error.
    n_in = int(train_cfg["model"]["in_channels"])
    channels = channels[:n_in]

    bs = int(train_cfg["train"]["batch_size"])
    nw = int(train_cfg["data"].get("num_workers", 4))

    collapse = bool(train_cfg["data"].get("collapse_classes", False))
    if collapse:
        from iceroute.data.collapse import describe

        log.warning("%s", describe())

    train_scenes = list(split["train"])
    if train_cfg["data"].get("use_split_extra", False):
        # Extra scenes extend TRAIN only. Val and demo stay bit-for-bit the
        # frozen split (SPEC §5.3), which is the whole reason the before/after
        # comparison means anything: if the yardstick moved too, it would not.
        extra_path = Path("configs/split_extra.yaml")
        if extra_path.exists():
            extra = yaml.safe_load(extra_path.read_text()).get("extra_train", [])
            new = [s for s in extra if s not in set(train_scenes)]
            train_scenes += new
            log.info(
                "Split esteso: %d scene di training (%d dal congelato + %d aggiuntive). "
                "Val e demo invariate.", len(train_scenes), len(split["train"]), len(new),
            )
        else:
            log.warning(
                "use_split_extra attivo ma %s non esiste: uso il solo split congelato.",
                extra_path,
            )

    train_ds = PatchDataset(cache, train_scenes, channels=channels, augment=True,
                            seed=train_cfg["seed"], collapse_classes=collapse)
    val_ds = PatchDataset(cache, split["val"], channels=channels, augment=False,
                          collapse_classes=collapse)

    log.info("Patch: %d train, %d val — canali %s", len(train_ds), len(val_ds), channels)

    common = {
        "num_workers": nw,
        "pin_memory": bool(train_cfg["data"].get("pin_memory", True)),
        "collate_fn": collate,
        "persistent_workers": nw > 0,
    }

    # Ice-margin balancing happens here rather than in the cache (D-017): the
    # cache keeps every valid patch, and the target fraction is reached by
    # weighting the draw. Validation is never weighted — its job is to measure
    # the real class distribution, not a rebalanced one.
    target = float(data_cfg["patches"].get("margin_fraction_target", 0.40))
    weights = train_ds.sampler_weights(target)
    if weights is None:
        log.warning(
            "Nessun peso di campionamento (cache senza flag di margine, o un gruppo "
            "vuoto): uso lo shuffle semplice. Il %.0f%% di margine NON è garantito.",
            100 * target,
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True, drop_last=True, **common
        )
    else:
        from torch.utils.data import WeightedRandomSampler

        log.info(
            "Campionamento pesato: %d patch in cache, %.1f%% con margine → "
            "target %.0f%% per batch",
            len(train_ds), 100 * train_ds.margin_fraction(), 100 * target,
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, sampler=sampler, drop_last=True, **common
        )

    return (
        train_loader,
        DataLoader(val_ds, batch_size=int(train_cfg["eval"].get("batch_size", bs)), shuffle=False, **common),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, n_classes: int) -> tuple[float, ConfusionMatrix]:
    model.eval()
    cm = ConfusionMatrix(n_classes)
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            out = criterion(logits, y)
        total += float(out["loss"].detach()) * x.size(0)
        n += x.size(0)
        cm.update(y, logits.argmax(dim=1))
    return (total / n if n else float("nan")), cm


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, *, base_lr, sched, step0, grad_clip
) -> tuple[float, float, float, int]:
    model.train()
    tot = ce_tot = dice_tot = 0.0
    n = 0
    step = step0

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        for g in optimizer.param_groups:
            g["lr"] = base_lr * sched(step)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x)
            out = criterion(logits, y)

        scaler.scale(out["loss"]).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        tot += float(out["loss"].detach()) * bs
        ce_tot += float(out["ce"]) * bs
        dice_tot += float(out["dice"]) * bs
        n += bs
        step += 1

    return tot / max(n, 1), ce_tot / max(n, 1), dice_tot / max(n, 1), step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Training U-Net per segmentazione SOD.")
    ap.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    ap.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    ap.add_argument("--epochs", type=int, default=None, help="override, utile per smoke test")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-batches", type=int, default=None, help="smoke test")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = yaml.safe_load(args.config.read_text())
    data_cfg = load_config(args.data_config)
    split = yaml.safe_load(Path(data_cfg.get("split_file", "configs/split.yaml")).read_text())["split"]

    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("CUDA non disponibile: training su CPU, sarà lentissimo.")

    n_classes = int(cfg["model"]["n_classes"])
    model = build_model(cfg["model"]).to(device)
    log.info("%s", model.summary().replace("\n", "\n           "))

    stats_path = Path("artifacts/data_stats.json")
    criterion = build_loss(cfg, n_classes, stats_path).to(device)
    if criterion.class_weights.numel():
        log.info(
            "Pesi di classe (da %s): %s",
            stats_path.name,
            np.round(criterion.class_weights.cpu().numpy(), 2).tolist(),
        )
    else:
        log.warning("Nessun peso di classe: %s assente. Lancia 'make data'.", stats_path)

    train_loader, val_loader = build_loaders(data_cfg, cfg, split)

    tc = cfg["train"]
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["optimizer"]["lr"]),
        weight_decay=float(tc["optimizer"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(tc.get("amp", True)) and device.type == "cuda")

    epochs = int(tc["epochs"])
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup = int(tc["scheduler"].get("warmup_steps", 500))
    sched = lambda s: cosine_with_warmup(s, warmup=warmup, total=total_steps)  # noqa: E731

    out = cfg["output"]
    ckpt_path = Path(out["best_checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(Path(out["log_csv"]))

    es = tc.get("early_stopping", {})
    patience = int(es.get("patience", 10))
    best_miou, best_epoch, since_best = -1.0, -1, 0
    if cfg["data"].get("collapse_classes", False):
        from iceroute.data.collapse import NAMES as class_names
    else:
        class_names = {
            int(k): v["name"] for k, v in data_cfg["sod_classes"].items() if int(k) != 255
        }

    log.info("Inizio: %d epoche, %d step/epoca, batch %d, AMP %s",
             epochs, steps_per_epoch, tc["batch_size"], "on" if scaler.is_enabled() else "off")

    step = 0
    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        try:
            tr_loss, tr_ce, tr_dice, step = train_one_epoch(
                model, train_loader, criterion, opt, scaler, device,
                base_lr=float(tc["optimizer"]["lr"]), sched=sched, step0=step,
                grad_clip=float(tc.get("grad_clip_norm", 0)) or None,
            )
        except torch.cuda.OutOfMemoryError:
            new_bs = max(1, int(tc["batch_size"]) // 2)
            log.error("OOM con batch %d → riduco a %d e riprovo. "
                      "ANNOTARE in DECISIONS.md: il batch effettivo non è quello in config.",
                      tc["batch_size"], new_bs)
            tc["batch_size"] = new_bs
            torch.cuda.empty_cache()
            train_loader, val_loader = build_loaders(data_cfg, cfg, split)
            continue

        val_loss, cm = validate(model, val_loader, criterion, device, n_classes)
        miou = cm.miou()
        peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0

        res = EpochResult(
            epoch=epoch, train_loss=tr_loss, train_ce=tr_ce, train_dice=tr_dice,
            val_loss=val_loss, val_miou=miou, val_accuracy=cm.accuracy(),
            val_binary_accuracy=cm.binary_accuracy(),
            lr=opt.param_groups[0]["lr"], seconds=time.monotonic() - t0, peak_vram_gb=peak,
        )
        logger.append(res)

        marker = ""
        if miou > best_miou:
            best_miou, best_epoch, since_best = miou, epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_miou": miou,
                    "class_names": class_names,
                },
                ckpt_path,
            )
            marker = "  ← best"
        else:
            since_best += 1

        log.info(
            "epoca %3d/%d  loss %.4f  val %.4f  mIoU %.4f  acc %.4f  gh/acqua %.4f  "
            "lr %.2e  %.0fs  VRAM %.1f GB%s",
            epoch, epochs, tr_loss, val_loss, miou, cm.accuracy(),
            cm.binary_accuracy(), res.lr, res.seconds, peak, marker,
        )

        if args.limit_batches:
            log.info("--limit-batches: smoke test concluso.")
            break

        if since_best >= patience:
            log.info("Early stopping: %d epoche senza miglioramento (best %.4f all'epoca %d).",
                     patience, best_miou, best_epoch)
            break

    # --- final report -----------------------------------------------------
    log.info("=" * 66)
    log.info("Miglior mIoU: %.4f (epoca %d) → %s", best_miou, best_epoch, ckpt_path)

    _, cm = validate(model, val_loader, criterion, device, n_classes)
    print("\n" + cm.format_table(class_names) + "\n")

    thresholds = cfg.get("eval", {})
    min_miou = float(thresholds.get("min_miou", 0.60))
    min_bin = float(thresholds.get("min_binary_accuracy", 0.90))
    collapsed = bool(cfg["data"].get("collapse_classes", False))
    if collapsed:
        # The 0.60 in SPEC §7.4 is a threshold on the SIX-class problem. Quoting
        # it against a three-class score would compare two different tasks, and
        # in the flattering direction. Whatever this run scores, it does not
        # clear that gate — it replaces it.
        log.warning(
            "mIoU %.3f sullo schema RIDOTTO a 3 classi. Non è confrontabile con la "
            "soglia SPEC §7.4 di %.2f, che vale per le 6 classi SOD: questo è un "
            "problema diverso e più facile. Il ripiego §7.4 è già stato attivato, "
            "quindi il numero da ridimensionare è quello nel pitch, non questo.",
            best_miou, min_miou,
        )
    elif best_miou < min_miou:
        log.warning(
            "mIoU %.3f sotto la soglia SPEC di %.2f. SPEC §7.4: se dopo 3 giorni non si "
            "arriva, ridurre a binario ghiaccio/acqua + una classe di ghiaccio spesso, e "
            "ridimensionare le affermazioni nel pitch. Non aggiustare il numero.",
            best_miou, min_miou,
        )
    if cm.binary_accuracy() < min_bin:
        log.warning("Accuratezza ghiaccio/acqua %.3f sotto la soglia di %.2f.", cm.binary_accuracy(), min_bin)

    summary_path = Path("artifacts/train_summary.json")
    summary_path.write_text(json.dumps({
        "best_val_miou": best_miou,
        "best_epoch": best_epoch,
        "seed": cfg["seed"],
        "batch_size_effective": tc["batch_size"],
        "n_parameters": model.param_count(),
        "final": cm.as_dict(class_names),
    }, indent=2))
    log.info("Riepilogo in %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
