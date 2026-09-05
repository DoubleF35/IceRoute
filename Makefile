# IceRoute — orchestrazione. Ogni target è un gate della SPEC.
#
# Regola: ogni target o fa la cosa, o fallisce con un messaggio che spiega
# esattamente cosa manca. Mai un successo silenzioso su codice non implementato.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
UV      := $(shell command -v uv 2>/dev/null)

# Scena usata da `make inspect` se non se ne passa un'altra:
#   make inspect SCENE=data/raw/train/<file>.nc
SCENE   ?=

# Guardia: quasi tutti i target richiedono l'ambiente. `setup` no, ovviamente.
define REQUIRE_VENV
	@test -x $(PY) || { \
		echo ""; \
		echo "  L'ambiente virtuale non esiste ancora."; \
		echo "  Esegui prima:  make setup"; \
		echo ""; \
		exit 1; \
	}
endef

# ---------------------------------------------------------------------------
# Aiuto
# ---------------------------------------------------------------------------

.PHONY: help
help:
	@echo ""
	@echo "  IceRoute — comandi disponibili"
	@echo ""
	@echo "  Ambiente"
	@echo "    make setup        crea .venv (Python 3.11), installa le dipendenze pinnate,"
	@echo "                      verifica CUDA  [Gate 0.1]"
	@echo ""
	@echo "  Dati"
	@echo "    make download     scarica il sottoinsieme AI4Arctic (ripartibile, in background)"
	@echo "    make inspect      report su una scena   SCENE=data/raw/train/<file>.nc  [Gate 0.2]"
	@echo "    make data         estrae e cachea le patch + statistiche  [Gate 1]"
	@echo ""
	@echo "  Modello"
	@echo "    make train        training U-Net"
	@echo "    make eval         mIoU, IoU per classe, matrice di confusione, baseline"
	@echo "    make quantize     export ONNX + INT8 + misure size/latenza/RAM  [Gate 2]"
	@echo ""
	@echo "  Rotte"
	@echo "    make route        geocodifica + POLARIS + A* sulle scene demo  [Gate 3]"
	@echo ""
	@echo "  Demo e pitch"
	@echo "    make precompute   genera data/demo/"
	@echo "    make report       artifacts/metrics.json + figure per il deck"
	@echo "    make demo         avvia la UI  [Gate 4]"
	@echo "    make video        registra il video di backup 90s"
	@echo ""
	@echo "  Qualità"
	@echo "    make test         pytest"
	@echo "    make dashboard    rigenera index.html: stato del progetto nel browser"
	@echo "    make status       a che punto siamo (file presenti, gate superati)"
	@echo "    make all          data -> train -> eval -> quantize -> route -> precompute -> report"
	@echo ""

# ---------------------------------------------------------------------------
# Ambiente  [Gate 0.1]
# ---------------------------------------------------------------------------

.PHONY: setup
setup:
	@test -n "$(UV)" || { \
		echo "uv non trovato. Installalo con:  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
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

# ---------------------------------------------------------------------------
# Dati
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
		echo "Download avviato in background (PID $$!). Log: artifacts/logs/download.log"

.PHONY: inspect
inspect:
	$(REQUIRE_VENV)
	@test -n "$(SCENE)" || { \
		echo ""; \
		echo "  Serve una scena. Uso:"; \
		echo "    make inspect SCENE=data/raw/train/<file>.nc"; \
		echo ""; \
		echo "  Scene disponibili:"; \
		ls -1 data/raw/train/*.nc 2>/dev/null | head -5 || echo "    (nessuna: lancia prima 'make download')"; \
		echo ""; \
		exit 1; \
	}
	$(PY) -m iceroute.data.inspect "$(SCENE)" --figure artifacts/figures/inspect_$$(basename "$(SCENE)" .nc).png

.PHONY: data
data:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.data.patches --config configs/data.yaml

# ---------------------------------------------------------------------------
# Modello
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

# ---------------------------------------------------------------------------
# Rotte
# ---------------------------------------------------------------------------

.PHONY: route
route:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.routing.run --config configs/routing.yaml

# ---------------------------------------------------------------------------
# Demo e pitch
# ---------------------------------------------------------------------------

.PHONY: precompute
precompute:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.demo.precompute --config configs/routing.yaml

.PHONY: dashboard
dashboard:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.pitch.dashboard
	@echo "  Apri index.html nel browser."

.PHONY: report
report:
	$(REQUIRE_VENV)
	$(PY) -m iceroute.pitch.metrics
	$(PY) -m iceroute.pitch.figures
	$(PY) -m iceroute.pitch.dashboard

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
# Qualità
# ---------------------------------------------------------------------------

.PHONY: test
test:
	$(REQUIRE_VENV)
	$(PYTEST) tests/ -v

.PHONY: status
status:
	$(REQUIRE_VENV)
	@$(PY) scripts/status.py

.PHONY: all
all: data train eval quantize route precompute report
	@echo ""
	@echo "  Pipeline completa. Verifica artifacts/metrics.json prima del pitch."
	@echo ""

.PHONY: clean-cache
clean-cache:
	rm -rf data/processed/*
	@echo "Cache patch svuotata. 'make data' la rigenera."
