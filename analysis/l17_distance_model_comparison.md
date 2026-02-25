# Distance Model Comparison: Our Simulator vs L17_533 Benchmark

## Summary

For instance `c10_8502` (Conventional layout), the L17_533 benchmark reports a best known objective of **234.0**, while our simulator computes **1186.0** for the same solution assignment. This 5x difference stems from fundamentally different routing models. This document traces the exact computation of both values.

## Instance c10_8502 at a Glance

| Property | Value |
|---|---|
| Layout | Conventional (220 pick locations, 2 depots) |
| Depot 0 (start) | [20, 5] |
| Depot 1 (end) | [50, 5] |
| Orders | 5 orders, 2 SKUs each = 10 picks |
| SKUs to slot | 1 (SKU 6) |
| Vehicles | 2, capacity 3 orders each |
| Distance metric | Manhattan |

### Solution Assignment (from `c10_8502_sol.json`)

| SKU | Location | Coordinates |
|-----|----------|-------------|
| 2 | 49 | [37, 24] |
| 3 | 153 | [37, 40] |
| 4 | 185 | [55, 44] |
| 5 | 182 | [51, 40] |
| 6 | 68 | [45, 24] |
| 7 | 55 | [43, 16] |
| 8 | 209 | [67, 52] |
| 9 | 193 | [61, 40] |
| 10 | 65 | [49, 16] |
| 11 | 50 | [33, 28] |

### Orders

| Order | SKUs | Pick Locations |
|-------|------|----------------|
| 1 | 2, 3 | [37,24], [37,40] |
| 2 | 4, 5 | [55,44], [51,40] |
| 3 | 6, 7 | [45,24], [43,16] |
| 4 | 8, 9 | [67,52], [61,40] |
| 5 | 10, 11 | [49,16], [33,28] |

---

## Benchmark Routing Model: Order Batching Problem (OBP)

The L17_533 benchmark evaluates assignments using an **Order Batching Problem** solver:

1. **Batch orders** across vehicles (2 vehicles, capacity 3 orders each)
2. **Solve TSP** for each vehicle's batch — find the shortest round-trip from a depot through all pick locations
3. **Sum** vehicle route costs = total objective

### Step-by-step computation (yielding 234.0)

**Optimal batching**: Vehicle 1 gets orders {1, 2, 4}, Vehicle 2 gets orders {3, 5}.

**Vehicle 1** — Orders {1, 2, 4} → SKUs {2, 3, 4, 5, 8, 9} → 6 picks:

Pick locations: [37,24], [37,40], [55,44], [51,40], [67,52], [61,40]

TSP round-trip from depot 1 [50,5]:
```
depot1[50,5] → loc65[49,16] is NOT in this batch
Best TSP tour from [50,5]:
  [50,5] → [37,24] → [37,40] → [51,40] → [55,44] → [67,52] → [61,40] → [50,5]
    = 32 + 16 + 14 + 8 + 20 + 18 + 46 = 154
```
Cost: **154**

**Vehicle 2** — Orders {3, 5} → SKUs {6, 7, 10, 11} → 4 picks:

Pick locations: [45,24], [43,16], [49,16], [33,28]

TSP round-trip from depot 1 [50,5]:
```
Best TSP tour from [50,5]:
  [50,5] → [49,16] → [43,16] → [45,24] → [33,28] → [50,5]
    = 12 + 6 + 10 + 16 + 40 = 84...
```

Actually the optimal TSP is 80 (different ordering). With 4 locations, brute-force gives:
```
Optimal: [50,5] → [49,16] → [45,24] → [33,28] → [43,16] → [50,5]
  = 12 + 12 + 16 + 22 + 18 = 80
```
Cost: **80**

**Total: 154 + 80 = 234** ✓

### Key characteristics

- **Batching**: multiple orders grouped per vehicle to share travel
- **TSP routing**: optimal tour through all picks in a batch (not sequential)
- **Round-trip**: depart from and return to **depot 1** only
- **No inter-pick overhead**: within a batch, travel is continuous

---

## Our Simulator Routing Model

Our `Simulator.simulate()` processes picks **individually and sequentially**:

1. For each pick (in timestamp order):
   - Travel: start_point (depot 0) → item_location → end_point (depot 1)
2. Between picks:
   - Return: end_point (depot 1) → start_point (depot 0)

### Step-by-step computation (yielding 1186.0)

Each pick is an independent trip. Manhattan distances from depot 0 [20,5] to location to depot 1 [50,5]:

| Pick | SKU | Location | d(depot0→loc) | d(loc→depot1) | Leg total |
|------|-----|----------|---------------|---------------|-----------|
| 1 | 2 | [37,24] | 36 | 32 | 68 |
| 2 | 3 | [37,40] | 52 | 48 | 100 |
| 3 | 4 | [55,44] | 74 | 44 | 118 |
| 4 | 5 | [51,40] | 66 | 36 | 102 |
| 5 | 6 | [45,24] | 44 | 24 | 68 |
| 6 | 7 | [43,16] | 34 | 18 | 52 |
| 7 | 8 | [67,52] | 94 | 64 | 158 |
| 8 | 9 | [61,40] | 76 | 46 | 122 |
| 9 | 10 | [65,16] | 40 | 12 | 52 |
| 10 | 11 | [33,28] | 36 | 40 | 76 |

**Sum of pick legs**: 68+100+118+102+68+52+158+122+52+76 = **916**

**Return trips** (depot1 → depot0 between picks): |50−20| + |5−5| = 30 each, 9 returns = **270**

**Total: 916 + 270 = 1186** ✓

---

## Side-by-side Comparison

| Aspect | Our Simulator | L17_533 Benchmark |
|--------|--------------|-------------------|
| **Result** | 1186.0 | 234.0 |
| **Pick grouping** | Individual (1 pick per trip) | Batched (multiple orders per vehicle) |
| **Routing** | Fixed: start → loc → end | TSP: optimal tour through batch |
| **Start/End** | Depot 0 → Depot 1 (asymmetric) | Round-trip from Depot 1 |
| **Return overhead** | 30 per inter-pick return × 9 = 270 | None (continuous tour) |
| **Vehicles** | 1 (implicit) | 2, capacity 3 orders each |

### What drives the 5x difference?

1. **No batching** (+largest factor): Our simulator makes 10 separate depot round-trips. The benchmark makes 2 tours total, sharing travel between co-located picks.

2. **No TSP optimization**: Within even a single trip, our simulator goes start→loc→end regardless of other picks. The benchmark finds the shortest tour.

3. **Return leg overhead**: 9 × 30 = 270 units of pure return travel that doesn't exist in the batched model.

---

## Implications for GNN Training

Despite the absolute distance difference, **our simulator is still valid for GNN training**:

1. **Relative ranking preservation**: If assignment A has lower total distance than assignment B under our routing, it likely also has lower distance under OBP routing. The ordinal ranking of assignments is what matters for the GNN to learn.

2. **Gradient signal**: The GNN learns which item→location mappings reduce travel distance. The magnitude differs, but the direction of improvement is the same.

3. **Limitation**: Our simulator cannot reproduce the exact benchmark objective. When reporting results, we should compare against our own simulation of the solution assignment (1186.0), not the benchmark's 234.0.

4. **Potential improvement**: To match the benchmark more closely, we could implement an OBP-aware simulator that batches orders and solves TSP routing. This would require:
   - Order batching logic (bin-packing across vehicles)
   - TSP solver (or nearest-neighbor heuristic) per batch
   - Vehicle capacity constraints from instance specs
