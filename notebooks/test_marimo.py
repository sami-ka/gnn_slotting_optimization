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
    from slotting_optimization.generator import DataGenerator
    from slotting_optimization.order_book import OrderBook
    from slotting_optimization.item_locations import ItemLocations
    from slotting_optimization.warehouse import Warehouse

    gen = DataGenerator()
    samples = gen.generate_samples(5, 20, 2, 4, n_samples=1, distances_fixed=True, seed=5)
    ob, il, w = samples[0]
    # each logical order id should appear between min and max times
    df = ob.to_df()
    return


if __name__ == "__main__":
    app.run()
