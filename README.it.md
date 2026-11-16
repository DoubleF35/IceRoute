# IceRoute — in breve

> Documentazione completa in inglese: [README.md](README.md).

**Segmentazione del ghiaccio marino a bordo satellite e rotte artiche vincolate da POLARIS,
dimensionate per un CubeSat.**

Una U-Net da 2 MB in INT8 legge una scena radar Sentinel-1, mappa il ghiaccio, calcola la rotta
più veloce che una data nave può percorrere secondo le regole IMO POLARIS e trasmette a terra
**il verdetto** invece dell'immagine: 148 KB invece di 55 MB, di cui la rotta occupa 431 byte.

🏆 **Primo premio — ESA EO MakerSpace Hackathon** (Frascati, giugno 2026)

## Numeri principali (test set ufficiale AI4Arctic, 20 scene)

| | |
|---|---|
| Segmentazione, 3 classi | mIoU **0,532** · accuratezza ghiaccio/acqua **0,906** |
| Stessa decisione operativa della carta del ghiaccio (POLARIS, PC6) | **89,0 %** dei pixel |
| Modello | 7,53 MB FP32 → **2,05 MB INT8**, nessuna perdita misurabile |
| Latenza per tile 256², 1 thread CPU | 93 ms FP32 → **40 ms INT8** |
| Compressione del downlink | **439:1** (mediana) |
| Calibrazione (ECE, test) | **16,4 % → 3,9 %** con una sola temperatura |
| Regola a rischio controllato (α = 0,1) | pixel non sicuri **5,6 % → 2,3 %**, senza riaddestrare ([D-043](docs/decisions.md#d-043)) |

## Avvio rapido

```bash
make setup && make fetch-demo && make demo
```

## Autore

Il concept è nato con il team **Space Eagle** all'ESA EO MakerSpace Hackathon. Il software di
questo repository è stato progettato e scritto da **Federico Fassio** (settembre–novembre 2026).
