# IceRoute

Segmentazione del ghiaccio marino a bordo satellite e calcolo della rotta artica ottimale
entro i limiti operativi POLARIS.

Un modello quantizzato INT8 gira su immagini SAR Sentinel-1, produce una mappa di ghiaccio,
calcola la rotta navale ottimale e downlinka **solo mask + rotta** invece della scena grezza.

---

## Partenza rapida

```bash
make setup      # Python 3.11 + dipendenze pinnate + verifica CUDA   [Gate 0.1]
make download   # sottoinsieme del dataset AI4Arctic (ripartibile)
make demo       # avvia la UI
```

`make` da solo elenca tutti i comandi.

**Per il pitch serve solo `make demo`.** Funziona senza rete: legge esclusivamente file
precomputati in `data/demo/`.

---

## Requisiti

| | |
|---|---|
| Python | 3.11 (installato automaticamente da `uv`) |
| GPU | NVIDIA con ≥ 6 GB VRAM per il training. La demo gira su CPU. |
| Disco | ≥ 60 GB liberi per il dataset |
| `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

Il training è stato sviluppato su RTX 4050 Laptop (6 GB). AMP è obbligatoria: senza,
va in OOM.

---

## Com'è fatto

```
scena SAR (HH, HV, angolo di incidenza)
        │
        ▼
   U-Net < 2M parametri                    src/iceroute/model/
        │  segmentazione stage of development
        ▼
   mask di ghiaccio  ──────────────────────┐
        │                                  │
        ▼                                  │
   geocodifica → griglia EPSG:3413 500 m   │   src/iceroute/routing/geocode.py
        │                                  │
        ▼                                  │
   RIO POLARIS (mask + classe della nave)  │   src/iceroute/routing/polaris.py
        │                                  │
        ▼                                  │
   RIO → velocità → costo in tempo         │   src/iceroute/routing/cost.py
        │                                  │
        ▼                                  ▼
   A* 16-connesso ──────────────► payload downlink
        │                          (mask compressa + rotta GeoJSON)
        ▼
   rotta + confronto con baseline geodetica
```

Il costo di attraversamento di una cella è **tempo**, non distanza. È la ragione per cui la
rotta più corta e la più veloce non coincidono, ed è quello che la demo mostra.

---

## Struttura

| Percorso | Cosa contiene |
|---|---|
| `src/iceroute/data/` | lettura netCDF, estrazione patch, ispezione |
| `src/iceroute/model/` | U-Net, training, valutazione, export ONNX, quantizzazione |
| `src/iceroute/routing/` | geocodifica, POLARIS, funzione di costo, A*, baseline |
| `src/iceroute/pitch/` | compressione misurata, aggregatore metriche, figure |
| `src/iceroute/demo/` | precomputazione e UI Streamlit |
| `configs/` | tutta la configurazione. Nessun parametro hardcodato nel codice. |
| `artifacts/` | output generati: modelli, figure, `metrics.json` |
| `data/polaris_rv.csv` | tabella Risk Value POLARIS, **versionata**, con la fonte per riga |
| `present.html` | modalità presentazione: una schermata per passo, `make present` (non in git) |
| `pitch/` | materiali per il palco: script parlato (IT, e EN per le domande), modifiche slide per slide, tabella delle classi |
| `demo.html` | la demo congelata in un file solo. Doppio clic, zero dipendenze (`make demo-static`) |
| `index.html` | stato del progetto nel browser (`make dashboard`) |

---

## Regole del progetto

Sono nella specifica (`ICEROUTE_BUILD_SPEC.md`), ma le tre che contano:

1. **Nessun numero inventato.** Ogni valore mostrato nella demo o nel deck o è misurato da
   uno script del repo, o ha una fonte citata. I valori non ancora risolti sono marcati
   `PROVVISORIO` e elencati in [DECISIONS.md](DECISIONS.md) §99.
2. **`DECISIONS.md` si aggiorna ad ogni scelta non ovvia.** Serve a rispondere alle domande
   tecniche senza rileggere il codice.
3. **Split per scena, mai per patch.** Patch della stessa scena in train e validation
   producono metriche gonfie che crollano in demo.

---

## Backup

`data/demo/` e `demo.html` sono gitignored perché grandi, ma sono **gli artefatti più
importanti del progetto**: sono ciò che fa girare la demo senza rete. Se muore il disco,
muoiono con lui.

```bash
make backup
```

Produce `~/iceroute-backup-<data>.tar.gz` (~77 MB) con `data/demo/`, `demo.html`,
`artifacts/models/` e `artifacts/metrics.json`. **Poi copialo altrove** — chiavetta o cloud.
Un backup accanto all'originale non è un backup.

---

## Dataset

**AI4Arctic Sea Ice Challenge Dataset**, versione `ready-to-train`, prodotto da DTU Space e
ESA Φ-lab per la competizione AutoICE.
Collection DOI: [`10.11583/DTU.c.6244065`](https://doi.org/10.11583/DTU.c.6244065).

Le etichette sono ice chart prodotti da analisti esperti secondo la classificazione WMO.
Non sono riproducibili a mano, ed è la ragione per cui si usa questo dataset invece di
scaricare prodotti Sentinel-1 grezzi.

Si scarica un sottoinsieme stratificato, non le 513 scene complete. La selezione è
**congelata** in `configs/data.yaml`: una volta scelta non si rigenera, altrimenti le
metriche di validazione non sono più confrontabili fra loro.

---

## Stato

`make status` stampa quali gate sono superati e cosa manca.
