# Read pytorch model checkpoint from pt file
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool


# Model architecture (must match the training notebook)
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
        edge_attr = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_j, edge_attr):
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))


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
        edge_attr = self.edge_mlp(torch.cat([x[row], x[col], edge_attr], dim=1))
        return x, edge_attr

    def message(self, x_j, edge_attr):
        return self.node_mlp(torch.cat([x_j, edge_attr], dim=1))


class GCNBlock(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        self.edge_then_node = EdgeThenNodeLayer(node_dim, edge_dim)
        self.node_then_edge = NodeThenEdgeLayer(node_dim, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        x, edge_attr = self.edge_then_node(x, edge_index, edge_attr)
        x, edge_attr = self.node_then_edge(x, edge_index, edge_attr)
        return x, edge_attr


class GraphRegressionModel(nn.Module):
    def __init__(self, hidden_dim, edge_dim, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
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
        # Initialize node features with small random values
        x = torch.randn(
            (data.num_nodes, self.hidden_dim), device=data.edge_index.device
        ) * 0.01

        edge_attr = self.edge_encoder(data.edge_attr)

        for layer in self.layers:
            x, edge_attr = layer(x, data.edge_index, edge_attr)

        graph_emb = global_add_pool(x, data.batch)
        out = self.regressor(graph_emb)
        return out.squeeze(-1)


# Load the checkpoint with CPU mapping (model was trained on GPU)
device = torch.device("cpu")
checkpoint = torch.load("final_model.pt", map_location=device, weights_only=False)

# Recreate the model architecture and load weights
model = GraphRegressionModel(hidden_dim=64, edge_dim=3, num_layers=5)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Get normalization parameters for denormalizing predictions
mean_y = checkpoint["mean_y"]
std_y = checkpoint["std_y"]

# Read test set
test_data = torch.load("test_dataset.pt", map_location=device, weights_only=False)

print(f"Model loaded successfully!")
print(f"Normalization params: mean_y={mean_y:.4f}, std_y={std_y:.4f}")
print(f"Test dataset size: {len(test_data)} samples")

# Run inference on test set
from torch_geometric.loader import DataLoader
import math

loader = DataLoader(test_data, batch_size=64, shuffle=False)

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in loader:
        preds_norm = model(batch)
        # Denormalize predictions and targets
        preds = preds_norm * std_y + mean_y
        targets = batch.y.squeeze() * std_y + mean_y
        all_preds.append(preds)
        all_targets.append(targets)

all_preds = torch.cat(all_preds)
all_targets = torch.cat(all_targets)

# Regression metrics
mae = (all_preds - all_targets).abs().mean().item()
rmse = math.sqrt(((all_preds - all_targets) ** 2).mean().item())
ss_res = ((all_targets - all_preds) ** 2).sum().item()
ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
r2 = 1 - ss_res / ss_tot
mape = ((all_preds - all_targets).abs() / all_targets.abs()).mean().item() * 100

print(f"\n=== Test Set Metrics ===")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"\nTarget range: [{all_targets.min().item():.1f}, {all_targets.max().item():.1f}]")
print(f"Pred range:   [{all_preds.min().item():.1f}, {all_preds.max().item():.1f}]")
