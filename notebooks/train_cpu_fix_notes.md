# Fix Notes: GNN Sensitivity Test (`train_cpu.py`)

## What was broken

The sensitivity test at the end of training always printed `diff = 0.0000`, even though the model was training fine. Three compounding bugs caused this.

---

### Bug 1 — Double RNG reset (the obvious one)

```python
# BEFORE (broken)
torch.manual_seed(42)
with torch.no_grad():
    orig_pred = model(sample).item()

...

torch.manual_seed(42)          # ← resets RNG to same state as before
with torch.no_grad():
    swap_pred = model(swapped_sample).item()
```

`torch.manual_seed(42)` before each call meant `torch.randn(...)` inside `forward()` produced **identical node feature matrices** for both graphs. The topology difference never had a chance to matter.

**Fix:** Remove the second `torch.manual_seed(42)`.

---

### Bug 2 — Random node features (the deeper one)

Even after fixing Bug 1, diff was still 0.0000. The reason:

```python
# BEFORE (broken)
x = torch.randn(num_nodes, self.hidden_dim, device=edge_index.device) * 0.01
```

All nodes started with small random noise — no structural identity. This causes a **permutation symmetry problem**:

- The sensitivity test swaps item A's location with item B's location
- Both items start with different random vectors (different RNG state now), so message passing produces different per-node embeddings after the swap
- But `global_add_pool` is a **sum** — and addition is commutative
- Swapping item A↔item B's locations just swaps which node gets which neighborhood signal, but the *sum* across all nodes ends up the same

So the model genuinely cannot distinguish the two graphs at the output level, regardless of what happens inside message passing.

---

### Bug 3 — No node identity in the architecture

The model had no mechanism to give individual nodes distinguishable initial features. Without unique node identities, items are interchangeable to the pooling layer.

---

## The fix

Replace random init with a learnable per-index node embedding:

```python
# In __init__:
self.node_embedding = nn.Embedding(256, hidden_dim)  # one embedding per node slot

# In forward(), replacing torch.randn:
nodes_per_graph = n_items + n_locs
node_ids = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device) % nodes_per_graph
x = self.node_embedding(node_ids)
```

Now node 0 (item 0) always starts with a different learned vector than node 1 (item 1). Swapping their location edges produces genuinely different neighborhood aggregations, and `global_add_pool` returns different values.

The `% nodes_per_graph` handles batched data where `num_nodes` is the total across all graphs in the batch (e.g. 32 samples × 12 nodes = 384). Without it, the embedding lookup goes out of range.

---

## Result after fix

```
Val loss: 0.0011   (was ~1.19)
Sensitivity diff: 0.4114   (was 0.0000)
SUCCESS: Model is sensitive to assignment changes!
```

The dramatic drop in val loss is also meaningful — the model can now actually learn which assignments are better, because nodes have identity.
