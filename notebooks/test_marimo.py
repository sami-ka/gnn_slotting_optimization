import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import polars as pl

    import torch
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 1, 1, 2],
                               [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data
    return Data, data, torch


@app.cell
def _(Data, torch):

    edge_index1 = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x1 = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    y1 = torch.tensor([10.2], dtype=torch.float)

    data1 = Data(x=x1, edge_index=edge_index1.t().contiguous(), y=y1)
    data1
    return


@app.cell
def _(data, torch):
    device = torch.device('cpu')
    _data = data.to(device)
    return


@app.cell
def _():
    # Add project root to sys.path (safe to paste in a notebook cell)
    from pathlib import Path
    import sys

    root = Path.cwd()
    # climb parents until we find the package folder
    while not (root / "slotting_optimization").exists() and root.parent != root:
        root = root.parent

    sys.path.insert(0, str(root))
    print("Added to sys.path:", root)
    return


@app.cell
def _():
    from slotting_optimization.generator import DataGenerator
    from slotting_optimization.order_book import OrderBook
    from slotting_optimization.item_locations import ItemLocations
    from slotting_optimization.warehouse import Warehouse

    gen = DataGenerator()
    samples = gen.generate_samples(2, 2, 20, 1, 4, n_samples=1, distances_fixed=True, seed=5)
    ob, il, w = samples[0]
    # each logical order id should appear between min and max times
    df = ob.to_df()

    return il, ob, w


@app.cell
def _(il, ob, w):
    from slotting_optimization.simulator import build_matrices_fast

    loc_mat, seq_mat, item_loc_mat, locs, items = build_matrices_fast(ob, il, w)

    return item_loc_mat, loc_mat, locs, seq_mat


@app.cell
def _(locs):
    import numpy as np

    nb_loc = len(locs)
    return (np,)


@app.cell
def _(loc_mat):
    loc_mat
    return


@app.cell
def _(item_loc_mat):
    item_loc_mat
    return


@app.cell
def _(item_loc_mat, loc_mat, np, seq_mat):
    np.concat([np.concat([loc_mat,item_loc_mat]), np.concat([item_loc_mat.T, seq_mat])],axis=1)
    return


app._unparsable_cell(
    r"""

        edge_index_ = torch.tensor([[0, 1],
                                   [1, 0],
                                   [1, 2],
                                   [2, 1]], dtype=torch.long)
        x_loc_item = torch.tensor([[1]*], dtype=torch.float)
        y1 = torch.tensor([10.2], dtype=torch.float)
    
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
