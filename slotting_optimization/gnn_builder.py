"""GNN builder functions for converting combined matrices to PyTorch Geometric graphs."""

from __future__ import annotations
from typing import Tuple, Optional, Callable, TYPE_CHECKING

import numpy as np
import torch
from torch_geometric.data import Data

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
    combined_matrix: np.ndarray,
    metadata: dict,
    validate_nan: bool = True,
    simulator: Optional[Callable] = None,
    order_book: Optional['OrderBook'] = None,
    item_locations: Optional['ItemLocations'] = None,
    warehouse: Optional['Warehouse'] = None
) -> Data:
    """Build sparse graph with edges only where matrix[i,j] != 0.

    Creates a directed graph from the combined matrix, including edges only for
    non-zero matrix values. Excludes self-loops (diagonal) and the bottom-right
    quadrant (location→item connections, which are always zero by design).

    Args:
        combined_matrix: Combined matrix from build_combined_matrix(), shape (I+L, I+L)
        metadata: Metadata dictionary with keys: n_items, n_locs, n_storage, items, locs
        validate_nan: If True, raise ValueError if any NaN values detected
        simulator: Optional callable that takes (order_book, warehouse, item_locations)
                   and returns (total_distance, per_order_distances).
                   Use Simulator().simulate or Simulator().simulate_sparse_matrix
        order_book: OrderBook for simulation (required if simulator provided)
        item_locations: ItemLocations for simulation (required if simulator provided)
        warehouse: Warehouse for simulation (required if simulator provided)

    Returns:
        torch_geometric.data.Data object with:
            - edge_index: [2, num_edges] tensor of edge indices
            - edge_attr: [num_edges, 1] tensor of edge weights
            - num_nodes: Total number of nodes (n_items + n_locs)
            - n_items, n_locs, n_storage, items_list, locs_list: Metadata attributes
            - y: [1] tensor containing total_distance (if simulator provided)

    Raises:
        KeyError: If required metadata key is missing
        ValueError: If matrix shape doesn't match metadata, NaN detected,
                    or incomplete simulation parameters

    Example:
        >>> # Without simulation target
        >>> combined, metadata = build_combined_matrix(ob, il, w)
        >>> data = build_graph_sparse(combined, metadata)
        >>> print(data.edge_index.shape)
        torch.Size([2, 87])

        >>> # With simulation target
        >>> from slotting_optimization.simulator import Simulator
        >>> sim = Simulator()
        >>> data = build_graph_sparse(
        ...     combined, metadata,
        ...     simulator=sim.simulate,
        ...     order_book=ob,
        ...     item_locations=il,
        ...     warehouse=w
        ... )
        >>> print(f"Target distance: {data.y.item()}")
        Target distance: 42.5
    """
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
        # Validate all simulation parameters provided
        if order_book is None or item_locations is None or warehouse is None:
            raise ValueError(
                "When simulator is provided, order_book, item_locations, "
                "and warehouse must all be provided"
            )

        # Run simulation
        total_distance, per_order = simulator(order_book, warehouse, item_locations)

        # Add to Data object as graph-level target
        data.y = torch.tensor([total_distance], dtype=torch.float)

    return data


def build_graph_dense(
    combined_matrix: np.ndarray,
    metadata: dict,
    validate_nan: bool = True,
    simulator: Optional[Callable] = None,
    order_book: Optional['OrderBook'] = None,
    item_locations: Optional['ItemLocations'] = None,
    warehouse: Optional['Warehouse'] = None
) -> Data:
    """Build dense graph with edges for all non-diagonal positions.

    Creates a directed graph from the combined matrix, including edges for all
    matrix positions (including zeros). Excludes self-loops (diagonal) and the
    bottom-right quadrant (location→item connections, which are always zero by design).

    Args:
        combined_matrix: Combined matrix from build_combined_matrix(), shape (I+L, I+L)
        metadata: Metadata dictionary with keys: n_items, n_locs, n_storage, items, locs
        validate_nan: If True, raise ValueError if any NaN values detected
        simulator: Optional callable that takes (order_book, warehouse, item_locations)
                   and returns (total_distance, per_order_distances).
                   Use Simulator().simulate or Simulator().simulate_sparse_matrix
        order_book: OrderBook for simulation (required if simulator provided)
        item_locations: ItemLocations for simulation (required if simulator provided)
        warehouse: Warehouse for simulation (required if simulator provided)

    Returns:
        torch_geometric.data.Data object with:
            - edge_index: [2, num_edges] tensor of edge indices
            - edge_attr: [num_edges, 1] tensor of edge weights (can include 0.0)
            - num_nodes: Total number of nodes (n_items + n_locs)
            - n_items, n_locs, n_storage, items_list, locs_list: Metadata attributes
            - y: [1] tensor containing total_distance (if simulator provided)

    Raises:
        KeyError: If required metadata key is missing
        ValueError: If matrix shape doesn't match metadata, NaN detected,
                    or incomplete simulation parameters

    Example:
        >>> # Without simulation target
        >>> combined, metadata = build_combined_matrix(ob, il, w)
        >>> data = build_graph_dense(combined, metadata)
        >>> print(data.edge_index.shape)
        torch.Size([2, 156])

        >>> # With simulation target
        >>> from slotting_optimization.simulator import Simulator
        >>> sim = Simulator()
        >>> data = build_graph_dense(
        ...     combined, metadata,
        ...     simulator=sim.simulate,
        ...     order_book=ob,
        ...     item_locations=il,
        ...     warehouse=w
        ... )
        >>> print(f"Target distance: {data.y.item()}")
        Target distance: 42.5
    """
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
        # Validate all simulation parameters provided
        if order_book is None or item_locations is None or warehouse is None:
            raise ValueError(
                "When simulator is provided, order_book, item_locations, "
                "and warehouse must all be provided"
            )

        # Run simulation
        total_distance, per_order = simulator(order_book, warehouse, item_locations)

        # Add to Data object as graph-level target
        data.y = torch.tensor([total_distance], dtype=torch.float)

    return data
