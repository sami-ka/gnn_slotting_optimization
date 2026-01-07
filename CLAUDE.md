# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GNN-based slotting optimization project for warehouse order preparation. The core simulates order picking travel distance to optimize item placement in warehouse locations. It combines discrete-event simulation with graph neural network capabilities (PyTorch Geometric) for learning optimal slotting strategies.

## Development Commands

### Environment Setup
This project uses `uv` for dependency management (Python 3.13+):
```bash
# Install dependencies (creates .venv automatically)
uv sync

# Add a new dependency
uv add <package-name>
```

### Testing
```bash
# Run all tests
uv run pytest ./tests

# Run specific test file
uv run pytest tests/test_simulator.py

# Run with verbose output
uv run pytest -v

# Run performance benchmarks
uv run pytest tests/test_simulator_perf.py -v
```

### Code Quality
```bash
# Format and lint code with ruff
uv run ruff check .
uv run ruff format .
```

### Notebooks
The project uses Marimo for interactive notebooks:
```bash
# Run marimo notebook
marimo edit notebooks/test_marimo.py
```

## Development Workflow

### Test-First Development (Required)

This project follows strict test-first development. **Always write tests before implementing features.**

**Before any implementation:**
1. **Challenge the user on test cases**: Ask specific questions about what should be tested:
   - What are the expected inputs and outputs?
   - What edge cases should be covered?
   - What error conditions need handling?
   - Are there performance requirements to validate?

2. **Write tests first**: Create test cases covering:
   - **Happy path**: Normal, expected usage
   - **Edge cases**: Boundary conditions, empty inputs, single items, large datasets
   - **Error handling**: Invalid inputs, missing data, constraint violations
   - **Integration**: How components interact (e.g., OrderBook → Simulator → results)
   - **Performance**: When relevant (use `test_*_perf.py` pattern for benchmarks)

3. **Implementation only after tests exist**: Do not implement functionality until test cases are written and the user has approved the test plan.

**Example test planning questions:**
- "For this new validation function, should it raise an exception or return False for invalid inputs?"
- "What should happen when the input DataFrame is empty?"
- "Do we need to test performance for datasets with >10k items?"
- "Should this work with both ISO timestamp strings and epoch timestamps?"

## Architecture Overview

### Core Simulation Components

**Data Models** (`slotting_optimization/models.py`):
- `Order`: Represents a single item pick (order_id, item_id, timestamp)
- `ItemLocation`: Maps items to warehouse locations

**Container Classes**:
- `OrderBook` (`order_book.py`): Manages orders in Polars DataFrame, handles CSV I/O, time-based filtering
- `ItemLocations` (`item_locations.py`): Manages item→location mapping with constraint that each location holds exactly one item type
- `Warehouse` (`warehouse.py`): Stores locations, start/end points, and directed distance mappings between location pairs

**Simulator** (`simulator.py`):
The `Simulator` class computes total travel distance for order sequences:
- Orders processed in timestamp order
- Each order: travel start → item_location → end
- Between orders: return end → start
- Two implementations:
  - `simulate()`: Polars-based with join operations
  - `simulate_sparse_matrix()`: Scipy sparse matrix approach for large-scale scenarios

**Matrix Conversion** (`simulator.py`):
Functions `build_matrices()` and `build_matrices_fast()` convert simulation data into three matrices for GNN input:
- `loc_mat` (L×L): Distance between warehouse locations
- `seq_mat` (I×I): Item-to-item sequence counts from orders
- `item_loc_mat` (I×L): Binary assignment matrix (item i at location j)

**Location Ordering**: The `locs` list returned by matrix functions is ordered as:
`[storage_location_0, ..., storage_location_n, start_point, end_point]`

This enables easy slicing for GNN processing:
```python
loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)
n_storage = len(locs) - 2  # All except start and end
storage_distances = loc_mat[:n_storage, :n_storage]
start_to_all = loc_mat[n_storage, :]
end_to_all = loc_mat[n_storage + 1, :]
```

The `_fast` variant uses vectorized Polars operations with window functions and NumPy advanced indexing for performance.

**Data Generation** (`generator.py`):
`DataGenerator.generate_samples()` creates synthetic (OrderBook, ItemLocations, Warehouse) tuples for testing and experimentation.

### Key Architectural Patterns

1. **Polars-First Data Processing**: All tabular operations use Polars DataFrames for performance. Data flows through `.to_df()` → process → aggregate pattern.

2. **Sparse Matrix Optimization**: Large-scale simulations use scipy.sparse CSR matrices to avoid materializing full distance tables.

3. **Constraint Enforcement**: ItemLocations enforces one-item-per-location invariant at construction time via `from_records()`.

4. **Private Attribute Access**: `build_matrices_fast()` accesses `warehouse._distance_map` directly (line 246 in simulator.py) to avoid O(L²) iteration - this is an intentional optimization.

5. **Test Organization**:
   - `tests/conftest.py` adds project root to sys.path
   - Separate perf tests (`test_simulator_perf.py`) for benchmarking
   - Sparse matrix tests (`test_simulator_sparse.py`) verify equivalence between implementations

### GNN Integration (Planned)

The main.py currently contains a basic GCN example using PyTorch Geometric's Cora dataset. The intended architecture will:
- Use matrix outputs from `build_matrices()` as graph features
- Train GNN to predict optimal item→location assignments
- Optimize for minimizing total travel distance from simulation

## Important Implementation Notes

- **Distance Map**: Warehouse stores directed distances as `(from_id, to_id) → float`. Missing distances raise ValueError during simulation.
- **Timestamp Handling**: OrderBook parses ISO 8601 strings to Polars Datetime. Order.parse_timestamp() supports ISO strings, epoch timestamps, and datetime objects.
- **Item Uniqueness**: An item_id can only be assigned to one location_id. Violations raise ValueError in ItemLocations.from_records().
- **Sample Data**: Test fixtures in `slotting_optimization/data/` provide small examples (sample_orders.csv, sample_item_locations.csv).

## Performance Considerations

- Use `simulate_sparse_matrix()` for >10k orders
- Use `build_matrices_fast()` for matrix conversion (5-10x faster than `build_matrices()`)
- Benchmarks in `scripts/benchmark_*.py` measure simulation and matrix performance
