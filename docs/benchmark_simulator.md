# Simulator Micro-benchmark
**Problem size:** 5000 orders, 1000 locations
## Summary
| Implementation | mean (s) | stdev (s) | min (s) | median (s) | reps |
|---|---:|---:|---:|---:|---:|
| baseline | 0.027517 | 0.001284 | 0.026203 | 0.027803 | 5 |
| sparse   | 0.022118 | 0.010029 | 0.017320 | 0.017825 | 5 |

**Speedup (baseline / sparse):** 1.244x
## Raw timings (s)
Baseline times:
- 0.027920
- 0.026341
- 0.026203
- 0.029316
- 0.027803

Sparse times:
- 0.040053
- 0.017913
- 0.017825
- 0.017477
- 0.017320
