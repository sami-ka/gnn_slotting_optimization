import marimo

__generated_with = "0.18.4"
app = marimo.App()


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
    samples = gen.generate_samples(200, 200, 1000, 1, 10, n_samples=1000, distances_fixed=True, seed=5)
    return (samples,)


@app.cell
def _(samples):
    from slotting_optimization.gnn_builder import build_graph_sparse
    from slotting_optimization.simulator import Simulator
    list_data = []
    for (ob, il, w) in samples:
        g_data = build_graph_sparse(
        order_book=ob,
        item_locations=il,
        warehouse=w,
        simulator=Simulator().simulate
    )
        list_data.append(g_data)

    return (list_data,)


@app.cell
def _(list_data):
    len(list_data)
    return


@app.cell
def _(list_data):
    from torch_geometric.data import InMemoryDataset
    class CustomDataset(InMemoryDataset):
        def __init__(self, listOfDataObjects):
            super().__init__()
            self.data, self.slices = self.collate(listOfDataObjects)

        def __len__(self):
            return len(self.slices)

        def __getitem__(self, idx):
            sample = self.get(idx)
            return sample
    dataset = CustomDataset(list_data)
    return (dataset,)


@app.cell
def _(dataset):
    dataset.collate
    return


@app.cell
def _(list_data):
    import torch
    torch.manual_seed(12345)

    train_dataset = list_data[:800]
    test_dataset = list_data[800:]

    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return test_loader, torch, train_loader


@app.cell
def _(train_loader):
    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()
    return (data,)


@app.cell
def _():
    from torch import nn
    from torch_geometric.nn import MessagePassing, global_add_pool
    return MessagePassing, global_add_pool, nn


@app.cell
def _(MessagePassing, nn, torch):
    class EdgeThenNodeLayer(MessagePassing):
        def __init__(self, node_dim, edge_dim):
            super().__init__(aggr="add")
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim),
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim + edge_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
            )

        def forward(self, x, edge_index, edge_attr):
            row, col = edge_index
            edge_attr = self.edge_mlp(
                torch.cat([x[row], x[col], edge_attr], dim=1)
            )
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return x, edge_attr

        def message(self, x_j, edge_attr):
            return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))
    return (EdgeThenNodeLayer,)


@app.cell
def _(MessagePassing, nn, torch):
    class NodeThenEdgeLayer(MessagePassing):
        def __init__(self, node_dim, edge_dim):
            super().__init__(aggr="add")
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim + edge_dim, node_dim),
                nn.ReLU(),
                nn.Linear(node_dim, node_dim),
            )
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * node_dim + edge_dim, edge_dim),
                nn.ReLU(),
                nn.Linear(edge_dim, edge_dim),
            )

        def forward(self, x, edge_index, edge_attr):
            x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            row, col = edge_index
            edge_attr = self.edge_mlp(
                torch.cat([x[row], x[col], edge_attr], dim=1)
            )
            return x, edge_attr

        def message(self, x_j, edge_attr):
            return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))
    return (NodeThenEdgeLayer,)


@app.cell
def _(EdgeThenNodeLayer, NodeThenEdgeLayer, nn):
    class GCNBlock(nn.Module):
        def __init__(self, node_dim, edge_dim):
            super().__init__()
            self.edge_then_node = EdgeThenNodeLayer(node_dim, edge_dim)
            self.node_then_edge = NodeThenEdgeLayer(node_dim, edge_dim)

        def forward(self, x, edge_index, edge_attr):
            x, edge_attr = self.edge_then_node(x, edge_index, edge_attr)
            x, edge_attr = self.node_then_edge(x, edge_index, edge_attr)
            return x, edge_attr
    return (GCNBlock,)


@app.cell
def _(GCNBlock, global_add_pool, nn, torch):
    class GraphRegressionModel(nn.Module):
        def __init__(self, num_nodes, edge_dim, hidden_dim, num_layers):
            super().__init__()

            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

            self.layers = nn.ModuleList(
                [GCNBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )

            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, data):
            node_ids = torch.arange(data.num_nodes, device=data.edge_index.device)
            x = self.node_embedding(node_ids)
            edge_attr = self.edge_encoder(data.edge_attr)

            for layer in self.layers:
                x, edge_attr = layer(x, data.edge_index, edge_attr)

            graph_emb = global_add_pool(x, data.batch)
            out = self.regressor(graph_emb)
            return out.squeeze(-1)
    return (GraphRegressionModel,)


@app.cell
def _(GraphRegressionModel, data):
    model = GraphRegressionModel(num_nodes=data.num_nodes,
                edge_dim=data.num_edge_features,
                hidden_dim=3,
                num_layers=3)
    print(model)
    return (model,)


@app.cell
def _(model, test_loader, torch, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
             out = model(data)  # Perform a single forward pass.
             loss = criterion(out, data.y)  # Compute the loss.
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            pred = model(data)
            # compute MSE
            correct += ((pred - data.y) ** 2).sum().item()
        return correct / len(loader.dataset)  # MSE

    for epoch in range(1, 10):
        train()
        train_mse = test(train_loader)
        test_mse = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
