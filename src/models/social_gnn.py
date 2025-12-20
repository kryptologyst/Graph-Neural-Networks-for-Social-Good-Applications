"""Enhanced GNN models for social good applications."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import TopKPooling


class GCNLayer(nn.Module):
    """Enhanced GCN layer with normalization and residual connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        normalization: str = "batch",
        residual: bool = False,
    ):
        """Initialize GCN layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            dropout: Dropout probability.
            normalization: Type of normalization ('batch', 'layer', 'none').
            residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual and (in_channels == out_channels)
        
        if normalization == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        elif normalization == "layer":
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            Updated node features.
        """
        residual = x if self.residual else None
        
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        if residual is not None:
            x = x + residual
        
        return x


class SocialRiskGCN(nn.Module):
    """Enhanced GCN for social risk prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        normalization: str = "batch",
        residual: bool = True,
    ):
        """Initialize the model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension (number of classes).
            num_layers: Number of GCN layers.
            dropout: Dropout probability.
            normalization: Type of normalization.
            residual: Whether to use residual connections.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            GCNLayer(input_dim, hidden_dim, dropout, normalization, False)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GCNLayer(hidden_dim, hidden_dim, dropout, normalization, residual)
            )
        
        # Output layer
        if num_layers > 1:
            self.layers.append(
                GCNLayer(hidden_dim, output_dim, 0.0, "none", False)
            )
        else:
            self.layers.append(
                GCNLayer(input_dim, output_dim, 0.0, "none", False)
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            Node predictions.
        """
        for layer in self.layers:
            x = layer(x, edge_index)
        
        return x


class SocialRiskSAGE(nn.Module):
    """GraphSAGE model for social risk prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        aggregator: str = "mean",
    ):
        """Initialize the model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of SAGE layers.
            dropout: Dropout probability.
            aggregator: Aggregation method ('mean', 'max', 'lstm').
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator))
        
        # Output layer
        if num_layers > 1:
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator))
        else:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            Node predictions.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            if i < len(self.layers) - 1:  # Not the last layer
                x = self.norm(x)
                x = F.relu(x)
                x = self.dropout(x)
        
        return x


class SocialRiskGAT(nn.Module):
    """Graph Attention Network for social risk prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        concat: bool = True,
    ):
        """Initialize the model.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden dimension.
            output_dim: Output dimension.
            num_layers: Number of GAT layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            concat: Whether to concatenate attention heads.
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout, concat=concat)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(
                    hidden_dim * num_heads if concat else hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=concat,
                )
            )
        
        # Output layer
        if num_layers > 1:
            self.layers.append(
                GATConv(
                    hidden_dim * num_heads if concat else hidden_dim,
                    output_dim,
                    heads=1,
                    dropout=0.0,
                    concat=False,
                )
            )
        else:
            self.layers.append(
                GATConv(input_dim, output_dim, heads=1, dropout=0.0, concat=False)
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            Node predictions.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            
            if i < len(self.layers) - 1:  # Not the last layer
                x = F.elu(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor) -> list:
        """Get attention weights for visualization.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            List of attention weights for each layer.
        """
        attention_weights = []
        
        for layer in self.layers:
            x, att = layer(x, edge_index, return_attention_weights=True)
            attention_weights.append(att)
            
            if layer != self.layers[-1]:  # Not the last layer
                x = F.elu(x)
        
        return attention_weights


class HierarchicalPooling(nn.Module):
    """Hierarchical pooling for graph-level tasks."""
    
    def __init__(self, in_channels: int, ratio: float = 0.5):
        """Initialize hierarchical pooling.
        
        Args:
            in_channels: Input feature dimension.
            ratio: Pooling ratio.
        """
        super().__init__()
        
        self.pool = TopKPooling(in_channels, ratio=ratio)
        self.global_pool = nn.ModuleList([
            global_mean_pool,
            global_max_pool,
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch assignment.
            
        Returns:
            Pooled graph representation.
        """
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        
        # Global pooling
        pooled_features = []
        for pool_fn in self.global_pool:
            pooled_features.append(pool_fn(x, batch))
        
        return torch.cat(pooled_features, dim=1)


def create_model(model_type: str, config: dict) -> nn.Module:
    """Create a model based on configuration.
    
    Args:
        model_type: Type of model ('gcn', 'sage', 'gat').
        config: Model configuration.
        
    Returns:
        Initialized model.
    """
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    
    if model_type == "gcn":
        return SocialRiskGCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            normalization=config.get("normalization", "batch"),
            residual=config.get("residual", True),
        )
    elif model_type == "sage":
        return SocialRiskSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            aggregator=config.get("aggregator", "mean"),
        )
    elif model_type == "gat":
        return SocialRiskGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=config.get("num_heads", 4),
            dropout=dropout,
            concat=config.get("concat", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
