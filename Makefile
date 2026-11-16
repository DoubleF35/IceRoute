# IceRoute — task runner.
#
# Rule: every target either does the thing, or fails with a message that says
# exactly what is missing. Never a silent success on unimplemented code.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

VENV    := .venv
PY      := $(VENV)/bin/python
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
UV      := $(shell command -v uv 2>/dev/null)

# Scene used by `make inspect`:
#   make inspect SCENE=data/raw/train/<file>.nc
SCENE   ?=

# Release that `make fetch-demo` downloads the demo bundle and weights from.
RELEASE ?= v1.0.0

# Guard: almost every target needs the environment. `setup` obviously does not.
define REQUIRE_VENV
	@test -x $(PY) || { \
		echo ""; \
		echo "  The virtual environment does not exist yet."; \
		echo "  Run first:  make setup"; \
		echo ""; \
		exit 1; \
	}
endef

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:
	@echo ""
	@echo "  IceRoute — available targets"
	@echo ""
	@echo "  Environment"
	@echo "    make setup        create .venv (Python 3.11), install pinned deps, check CUDA"
	@echo "    make fetch-demo   download the demo bundle + model weights from GitHub Releases"
	@echo ""
	@echo "  Data"
	@echo "    make download     download the AI4Arctic subset (resumable)"
	@echo "    make inspect      report on one scene   SCENE=data/raw/train/<file>.nc"
	@echo "    make data         extract and cache training patches + statistics"
	@echo ""
	@echo "  Model"
	@echo "    make train        train the U-Net"
	@echo "    make eval         mIoU, per-class IoU, confusion matrix, RIO error, baselines"
	@echo "    make quantize     ONNX export + INT8 quantization + size/latency/RAM"
	@echo "    make calibrate    temperature scaling + reliability diagram (ECE)"
	@echo "    make risk         safety vs. accuracy curves for risk-aware decision rules"
	@echo ""
	@echo "  Routing"
	@echo "    make route        geocoding + POLARIS + A* on the demo scenes"
	@echo ""
	@echo "  Demo and reporting"
	@echo "    make precompute   build data/demo/ (scenes, masks, routes, overlays)"
	@echo "    make report       artifacts/metrics.json, figures, audit tables, static demo"
	@echo "    make demo         Streamlit UI"
	@echo "    make demo-static  freeze the demo into demo.html (one file, no dependencies)"
	@echo "    make console      console.html: the same demo with a globe and real footprints"
	@echo "    make site         assemble site/ for GitHub Pages (console as index)"
	@echo ""
	@echo "  Quality"
	@echo "    make test         full test suite"
	@echo "    make test-fast    tests that need neither downloaded data nor a GPU (= CI)"
	@echo "    make lint         ruff check + format check"
	@echo "    make format       ruff format + autofix"
	@echo "    make dashboard    regenerate index.html: project status in the browser"
	@echo "    make status       which stages are done and what is missing"
	@echo "    make preflight    offline readiness check of every demo artifact"
	@echo "    make backup       archive what is not in git (demo bundle, weights)"
	@echo "    make all          data -> train -> eval -> quantize -> route -> precompute -> report"
	@echo ""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.PHONY: setup
setup:
	@test -n "$(UV)" || { \
		echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}
	uv python install 3.11
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e . --no-deps
	@$(MAKE) --no-print-directory check-cuda

.PHONY: check-cuda
check-cuda:
	$(REQUIRE_VENV)
	@$(PY) scripts/check_env.py

.PHONY: fetch-demo
fetch-demo:
	bash scripts/fetch_assets.sh $(RELEASE)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

.PHONY: download
download:
	$(REQUIRE_VENV)
	$(PY) scripts/download_subset.py --config configs/data.yaml

.PHONY: download-bg
download-bg:
	$(REQUIRE_VENV)
	@mkdir -p artifacts/logs
	@nohup $(PY) scripts/download_subset.py --config configs/data.yaml \
		>> artifacts/logs/download.log 2>&1 & \
		echo "Download started in the background (PID $$!). Log: artifacts/logs/download.log"

.PHONY: inspect
inspect:
	$(REQUIRE_VENV)
	@test -n "$(SCENE)" || { \
		echo ""; \
		echo "  A scene is required. Usage:"; \
		echo "    make inspect SCENE=data/raw/train/<file>.nc"; \
		echo ""; \
		echo "  Available scenes:"; \
		ls -1 data/raw/train/*.nc 2>/dev/null | head -5 || echo "    (none: run 'make download' first)"; \
		echo ""; \
		exit 1; \
	}
	$(PY) -m iceroute.data.inspect "$(SCENE)" --figure artifacts/figures/inspect_$$(basename "$(SCENE)" .nc).png

.PHONY: data
data:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.data.patches --config configs/data.yaml

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

.PHONY: train
train:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.model.train --config configs/train.yaml

.PHONY: eval
eval:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.model.evaluate --config configs/train.yaml

.PHONY: quantize
quantize:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.model.export --config configs/train.yaml
	$(PY) -m iceroute.model.quantize --config configs/train.yaml

.PHONY: calibrate
calibrate:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.model.calibration --config configs/train3_lovasz.yaml

.PHONY: risk
risk:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.model.risk --config configs/train3_lovasz.yaml

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

.PHONY: route
route:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.routing.run --config configs/routing.yaml

# ---------------------------------------------------------------------------
# Demo and reporting
# ---------------------------------------------------------------------------

.PHONY: precompute
precompute:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.demo.precompute --config configs/routing.yaml

.PHONY: dashboard
dashboard:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.report.dashboard
	@echo "  Open index.html in a browser."

.PHONY: report
report:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.report.metrics
	$(PY) -m iceroute.report.assumptions
	$(PY) -m iceroute.report.classes
	$(PY) -m iceroute.report.figures
	$(PY) -m iceroute.report.dashboard
	@$(MAKE) --no-print-directory demo-static

# The demo frozen into a single file: no Python, no server, no network.
.PHONY: demo-static
demo-static:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.web.standalone

.PHONY: console
console:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.web.console
	@echo ""
	@echo "  Double-click console.html. Drag the globe, scroll to zoom."
	@echo ""

# Static site for GitHub Pages: the console is the landing page, the frozen
# demo sits next to it. Built locally or by .github/workflows/pages.yml.
.PHONY: site
site: console demo-static
	rm -rf site && mkdir -p site
	cp console.html site/index.html
	cp demo.html site/demo.html
	touch site/.nojekyll
	@echo "  site/ ready ($$(du -sh site | cut -f1))."

.PHONY: demo
demo:
	$(REQUIRE_VENV)
	$(VENV)/bin/streamlit run src/iceroute/demo/app.py \
		--server.headless false \
		--server.fileWatcherType none \
		--browser.gatherUsageStats false

.PHONY: video
video:
	$(REQUIRE_VENV)
	bash scripts/record_demo.sh

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: test
test:
	$(REQUIRE_VENV)
	$(PYTEST) tests/ -v

.PHONY: test-fast
test-fast:
	$(REQUIRE_VENV)
	$(PYTEST) tests/ -q -m "not slow and not needs_data"

.PHONY: lint
lint:
	$(REQUIRE_VENV)
	$(RUFF) check .
	$(RUFF) format --check .

.PHONY: format
format:
	$(REQUIRE_VENV)
	$(RUFF) check --fix .
	$(RUFF) format .

.PHONY: status
status:
	$(REQUIRE_VENV)
	@$(PY) scripts/status.py

# Run with the network unplugged after a reboot, and after regenerating any
# artifact: it checks that the demo works from what is on disk.
.PHONY: preflight
preflight:
	$(REQUIRE_VENV)
	@$(PY) scripts/preflight.py

.PHONY: all
all: data train eval quantize route precompute report
	@echo ""
	@echo "  Full pipeline done. Check artifacts/metrics.json."
	@echo ""

# What is not in git and what the demo needs. See README, "Reproducing".
.PHONY: backup
backup:
	@out=$$HOME/iceroute-backup-$$(date +%Y%m%d-%H%M).tar.gz; \
	files=""; \
	for f in data/demo demo.html console.html artifacts/models artifacts/metrics.json; do \
		[ -e "$$f" ] && files="$$files $$f"; \
	done; \
	tar czf "$$out" $$files; \
	echo ""; \
	echo "  $$out  ($$(du -h "$$out" | cut -f1))"; \
	echo "  contains:$$files"; \
	echo ""; \
	echo "  Now copy it somewhere ELSE. A backup on the same disk is not a backup."; \
	echo ""

.PHONY: clean-cache
clean-cache:
	rm -rf data/processed/*
	@echo "Patch cache cleared. 'make data' rebuilds it."
