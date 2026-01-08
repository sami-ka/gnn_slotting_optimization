"""GNN builder functions for converting combined matrices to PyTorch Geometric graphs."""

from __future__ import annotations
from typing import Tuple, Optional, Callable, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data
from slotting_optimization.simulator import build_combined_matrix

if TYPE_CHECKING:
    from slotting_optimization.order_book import OrderBook
    from slotting_optimization.item_locations import ItemLocations
    from slotting_optimization.warehouse import Warehouse


def _validate_inputs(combined_matrix: np.ndarray, metadata: dict, validate_nan: bool) -> Tuple[int, int, int]:
    """Validate inputs and return dimensions.

    Args:
        combined_matrix: Combined matrix from build_combined_matrix()
        metadata: Metadata dictionary
        validate_nan: Whether to validate for NaN values

    Returns:
        Tuple of (n_items, n_locs, total_nodes)

    Raises:
        KeyError: If required metadata key is missing
        ValueError: If matrix shape doesn't match metadata or NaN detected
    """
    # Check metadata keys
    required_keys = ['n_items', 'n_locs', 'n_storage', 'items', 'locs']
    for key in required_keys:
        if key not in metadata:
            raise KeyError(f"metadata missing required key: '{key}'")

    # Extract dimensions
    I = metadata['n_items']
    L = metadata['n_locs']
    total_nodes = I + L

    # Validate shape
    if combined_matrix.shape != (total_nodes, total_nodes):
        raise ValueError(
            f"combined_matrix shape {combined_matrix.shape} doesn't match "
            f"metadata (expected {total_nodes}×{total_nodes})"
        )

    # NaN validation
    if validate_nan:
        nan_mask = np.isnan(combined_matrix)
        if nan_mask.any():
            nan_positions = np.argwhere(nan_mask)
            first_nan = nan_positions[0]
            raise ValueError(
                f"NaN detected in combined_matrix at position [{first_nan[0]}, {first_nan[1]}]. "
                f"Found {len(nan_positions)} total NaN values."
            )

    return I, L, total_nodes


def build_graph_sparse(
    order_book: 'OrderBook',
    item_locations: 'ItemLocations',
    warehouse: 'Warehouse',
    validate_nan: bool = True,
    use_fast: bool = True,
    simulator: Optional[Callable] = None
) -> Data:
    """Build sparse graph from warehouse order data.

    Creates a directed graph with edges only for non-zero matrix values.
    Internally computes the combined matrix from the provided data objects.

    Args:
        order_book: OrderBook containing order sequences
        item_locations: ItemLocations mapping items to warehouse locations
        warehouse: Warehouse with locations and distance mappings
        validate_nan: If True, raise ValueError if NaN values detected in matrix
        use_fast: If True, use optimized build_matrices_fast() for matrix computation
        simulator: Optional callable(order_book, warehouse, item_locations) -> (total_distance, per_order)
                   for adding simulation target as y attribute

    Returns:
        torch_geometric.data.Data object with:
            - edge_index: [2, num_edges] directed edge indices
            - edge_attr: [num_edges, 1] edge weights
            - num_nodes: n_items + n_locs
            - n_items, n_locs, n_storage, items_list, locs_list: metadata
            - y: [1] total_distance tensor (if simulator provided)

    Example:
        >>> from slotting_optimization.simulator import Simulator
        >>> sim = Simulator()
        >>> data = build_graph_sparse(ob, il, w, simulator=sim.simulate)
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
        >>> print(f"Target distance: {data.y.item()}")
    """
    # Compute combined matrix internally
    combined_matrix, metadata = build_combined_matrix(
        order_book, item_locations, warehouse, use_fast=use_fast
    )

    # Validate inputs
    I, L, total_nodes = _validate_inputs(combined_matrix, metadata, validate_nan)

    # Find all non-zero positions
    rows, cols = np.nonzero(combined_matrix)

    # Convert column indices to node indices
    # Matrix structure: [locations cols (0:L) | items cols (L:L+I)]
    # Node structure: items (0:I), locations (I:I+L)
    # So: col < L -> node = I + col (location node)
    #     col >= L -> node = col - L (item node)
    target_nodes = np.where(cols < L, I + cols, cols - L)

    # Filter: exclude diagonal (source == target) and bottom-right quadrant
    # Bottom-right quadrant: rows >= I (location nodes), cols >= L (item columns)
    mask = (rows != target_nodes) & ~((rows >= I) & (cols >= L))
    edge_sources = rows[mask]
    edge_targets = target_nodes[mask]
    edge_cols_filtered = cols[mask]

    # Extract edge values using original column indices
    edge_values = combined_matrix[edge_sources, edge_cols_filtered]

    # Create edge_index [2, num_edges]
    edge_index = torch.tensor(
        np.stack([edge_sources, edge_targets], axis=0),
        dtype=torch.long
    )

    # Create edge_attr [num_edges, 1]
    edge_attr = torch.tensor(
        edge_values.reshape(-1, 1),
        dtype=torch.float
    )

    # Create Data object with metadata
    # Note: Use items_list/locs_list to avoid conflicts with PyG's internal handling
    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=total_nodes,
        n_items=I,
        n_locs=L,
        n_storage=metadata['n_storage'],
        items_list=metadata['items'],
        locs_list=metadata['locs']
    )

    # Optional: Add simulation target
    if simulator is not None:
        # Run simulation (ob, il, w are now always available)
        total_distance, per_order = simulator(order_book, warehouse, item_locations)

        # Add to Data object as graph-level target
        data.y = torch.tensor([total_distance], dtype=torch.float)

    return data


def build_graph_dense(
    order_book: 'OrderBook',
    item_locations: 'ItemLocations',
    warehouse: 'Warehouse',
    validate_nan: bool = True,
    use_fast: bool = True,
    simulator: Optional[Callable] = None
) -> Data:
    """Build dense graph from warehouse order data.

    Creates a directed graph with edges for all matrix positions (including zeros).
    Excludes self-loops (diagonal) and the bottom-right quadrant.
    Internally computes the combined matrix from the provided data objects.

    Args:
        order_book: OrderBook containing order sequences
        item_locations: ItemLocations mapping items to warehouse locations
        warehouse: Warehouse with locations and distance mappings
        validate_nan: If True, raise ValueError if NaN values detected in matrix
        use_fast: If True, use optimized build_matrices_fast() for matrix computation
        simulator: Optional callable(order_book, warehouse, item_locations) -> (total_distance, per_order)
                   for adding simulation target as y attribute

    Returns:
        torch_geometric.data.Data object with:
            - edge_index: [2, num_edges] directed edge indices
            - edge_attr: [num_edges, 1] edge weights (can include 0.0)
            - num_nodes: n_items + n_locs
            - n_items, n_locs, n_storage, items_list, locs_list: metadata
            - y: [1] total_distance tensor (if simulator provided)

    Example:
        >>> from slotting_optimization.simulator import Simulator
        >>> sim = Simulator()
        >>> data = build_graph_dense(ob, il, w, simulator=sim.simulate)
        >>> print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
        >>> print(f"Target distance: {data.y.item()}")
    """
    # Compute combined matrix internally
    combined_matrix, metadata = build_combined_matrix(
        order_book, item_locations, warehouse, use_fast=use_fast
    )

    # Validate inputs
    I, L, total_nodes = _validate_inputs(combined_matrix, metadata, validate_nan)

    # Generate all possible (row, col) pairs in matrix space
    # Note: matrix has L+I columns, not total_nodes columns
    matrix_cols = L + I  # Same as total_nodes, but clearer
    all_rows = np.repeat(np.arange(total_nodes), matrix_cols)
    all_cols = np.tile(np.arange(matrix_cols), total_nodes)

    # Convert column indices to node indices
    # Matrix structure: [locations cols (0:L) | items cols (L:L+I)]
    # Node structure: items (0:I), locations (I:I+L)
    target_nodes = np.where(all_cols < L, I + all_cols, all_cols - L)

    # Filter: exclude diagonal (source == target) and bottom-right quadrant
    mask = (all_rows != target_nodes) & ~((all_rows >= I) & (all_cols >= L))
    edge_sources = all_rows[mask]
    edge_targets = target_nodes[mask]
    edge_cols_filtered = all_cols[mask]

    # Extract edge values (including zeros) using original column indices
    edge_values = combined_matrix[edge_sources, edge_cols_filtered]

    # Create edge_index [2, num_edges]
    edge_index = torch.tensor(
        np.stack([edge_sources, edge_targets], axis=0),
        dtype=torch.long
    )

    # Create edge_attr [num_edges, 1]
    edge_attr = torch.tensor(
        edge_values.reshape(-1, 1),
        dtype=torch.float
    )

    # Create Data object with metadata
    # Note: Use items_list/locs_list to avoid conflicts with PyG's internal handling
    data = Data(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=total_nodes,
        n_items=I,
        n_locs=L,
        n_storage=metadata['n_storage'],
        items_list=metadata['items'],
        locs_list=metadata['locs']
    )

    # Optional: Add simulation target
    if simulator is not None:
        # Run simulation (ob, il, w are now always available)
        total_distance, per_order = simulator(order_book, warehouse, item_locations)

        # Add to Data object as graph-level target
        data.y = torch.tensor([total_distance], dtype=torch.float)

    return data
