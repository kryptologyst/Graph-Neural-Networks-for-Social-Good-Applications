#!/usr/bin/env python3
"""
Project 434: Graph Neural Networks for Social Good Applications

This is a modernized, production-ready implementation of GNNs for social good applications.
The original simple implementation has been refactored into a comprehensive framework.

For the full implementation, see the src/ directory and run:
    python scripts/train.py
    streamlit run demo/streamlit_app.py

Original concept: Social risk prediction using GCN on contact networks
Modern implementation: Comprehensive GNN framework with multiple architectures,
evaluation metrics, interactive visualization, and production-ready features.
"""

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random
import numpy as np


def create_simple_social_network():
    """Create a simple social contact network (original implementation)."""
    # Generate synthetic social contact graph
    G = nx.barabasi_albert_graph(n=100, m=3)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
    
    # Node features (age, socioeconomic status, health, mobility)
    np.random.seed(42)
    features = np.random.rand(100, 4)
    x = torch.tensor(features, dtype=torch.float32)
    
    # Simulate risk labels (1 = high-risk, 0 = low-risk)
    labels = torch.zeros(100, dtype=torch.long)
    high_risk = random.sample(range(100), 20)
    labels[high_risk] = 1
    
    # Train/test split
    train_mask = torch.zeros(100, dtype=torch.bool)
    test_mask = torch.zeros(100, dtype=torch.bool)
    train_mask[:70] = True
    test_mask[70:] = True
    
    return Data(x=x, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask)


class SimpleRiskGCN(torch.nn.Module):
    """Simple GCN for social risk prediction (original implementation)."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 16)
        self.conv2 = GCNConv(16, 2)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


def train_simple_model():
    """Train the simple GCN model (original implementation)."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Create data
    data = create_simple_social_network()
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRiskGCN().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Training function
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Test function
    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = int((pred[data.test_mask] == data.y[data.test_mask]).sum())
        total = int(data.test_mask.sum())
        return correct / total
    
    # Training loop
    print("Training Simple GCN for Social Risk Prediction")
    print("=" * 50)
    
    for epoch in range(1, 31):
        loss = train()
        acc = test()
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
    
    print(f"\nFinal Test Accuracy: {test():.4f}")
    print("\nThis demonstrates the basic concept. For the full implementation with")
    print("multiple architectures, comprehensive evaluation, and interactive demos,")
    print("see the src/ directory and run:")
    print("  python scripts/train.py")
    print("  streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    train_simple_model()