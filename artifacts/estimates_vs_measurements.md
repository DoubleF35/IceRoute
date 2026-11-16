# Early design estimates vs. measurements

> Written by `make report`. Each row sets an early design-phase estimate
> (June 2026 prototype) next to the value measured on the current model.
> Do not edit by hand: the file is regenerated.

| ID | Early estimate | Measured | Verdict | Note |
|---|---|---|---|---|
| A-02 | 58 KB of INT8 weights | **2098 KB** | contradicted | Measured size is 2.05 MB. At 1.09 bytes/parameter, 58 KB is arithmetically impossible. |
| A-04 | 1–3% mIoU lost to INT8 | **+0.0046** | contradicted, favourably | The measured degradation is negligible: better than the estimate. |
| P-06 / compression | ~200:1 | **439:1** | supported | Measured value: 439:1. |
| mIoU | 6 stages of ice development | **0.532 on 3 classes** | contradicted | The model distinguishes 3 classes: open water, thin ice, thick ice. The 3-class fallback is active (docs/decisions.md D-020). |
| A-03 | 5.3 ms per tile (GPU@SAT projection) | **40 ms on CPU, 1 thread** | different quantity | CPU latency does not validate the GPU@SAT projection: they are two different things. The projection must be labelled as a projection. |
| safety | (no early estimate) | **41.3% permissive in the worst case** | undisclosed | The fraction of pixels where a route would cross forbidden ice. It belongs in any statement about the system, stated up front rather than found by someone else. |
