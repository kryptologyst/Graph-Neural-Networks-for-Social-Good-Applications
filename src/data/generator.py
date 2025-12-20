"""Data generation and processing utilities for social good applications."""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


class SocialGraphGenerator:
    """Generator for synthetic social contact graphs."""
    
    def __init__(self, config: Dict):
        """Initialize the generator.
        
        Args:
            config: Configuration dictionary with generation parameters.
        """
        self.config = config
        self.num_nodes = config["num_nodes"]
        self.num_features = config["num_features"]
        self.num_classes = config["num_classes"]
        self.high_risk_ratio = config["high_risk_ratio"]
        self.graph_type = config["graph_type"]
        self.m = config.get("m", 3)
    
    def generate_graph(self) -> nx.Graph:
        """Generate a synthetic social contact graph.
        
        Returns:
            networkx.Graph: Generated graph.
        """
        if self.graph_type == "barabasi_albert":
            G = nx.barabasi_albert_graph(self.num_nodes, self.m)
        elif self.graph_type == "erdos_renyi":
            p = 0.1  # Connection probability
            G = nx.erdos_renyi_graph(self.num_nodes, p)
        elif self.graph_type == "watts_strogatz":
            k = 6  # Each node connected to k nearest neighbors
            p = 0.3  # Rewiring probability
            G = nx.watts_strogatz_graph(self.num_nodes, k, p)
        elif self.graph_type == "small_world":
            G = nx.newman_watts_strogatz_graph(self.num_nodes, 6, 0.3)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        return G
    
    def generate_features(self, G: nx.Graph) -> torch.Tensor:
        """Generate node features representing social characteristics.
        
        Args:
            G: Input graph.
            
        Returns:
            torch.Tensor: Node features of shape (num_nodes, num_features).
        """
        features = []
        
        for node in G.nodes():
            # Age (normalized 0-1)
            age = random.uniform(0.2, 0.8)
            
            # Socioeconomic status (normalized 0-1)
            ses = random.uniform(0.0, 1.0)
            
            # Health condition (0-1, higher = better health)
            health = random.uniform(0.3, 1.0)
            
            # Mobility level (0-1, higher = more mobile)
            mobility = random.uniform(0.0, 1.0)
            
            # Community density (based on node degree)
            degree = G.degree(node)
            max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
            density = degree / max_degree
            
            # Geographic isolation (0-1, higher = more isolated)
            isolation = random.uniform(0.0, 1.0)
            
            # Access to resources (0-1, higher = better access)
            resources = random.uniform(0.0, 1.0)
            
            # Vulnerability score (combination of factors)
            vulnerability = (1 - health) * 0.3 + (1 - ses) * 0.2 + isolation * 0.2 + (1 - resources) * 0.3
            
            # Combine all features
            node_features = [age, ses, health, mobility, density, isolation, resources, vulnerability]
            
            # Pad or truncate to match required number of features
            if len(node_features) < self.num_features:
                node_features.extend([0.0] * (self.num_features - len(node_features)))
            else:
                node_features = node_features[:self.num_features]
            
            features.append(node_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def generate_labels(self, G: nx.Graph, features: torch.Tensor) -> torch.Tensor:
        """Generate risk labels based on graph structure and features.
        
        Args:
            G: Input graph.
            features: Node features.
            
        Returns:
            torch.Tensor: Binary risk labels (0=low risk, 1=high risk).
        """
        labels = torch.zeros(self.num_nodes, dtype=torch.long)
        
        # Calculate risk scores based on features and graph structure
        risk_scores = []
        
        for i, node in enumerate(G.nodes()):
            # Feature-based risk (using vulnerability feature)
            feature_risk = features[i, -1].item()  # vulnerability score
            
            # Graph-based risk (centrality measures)
            degree = G.degree(node)
            betweenness = nx.betweenness_centrality(G)[node]
            closeness = nx.closeness_centrality(G)[node]
            
            # Combine risks
            graph_risk = (degree * 0.4 + betweenness * 0.3 + closeness * 0.3)
            graph_risk = graph_risk / max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
            
            total_risk = feature_risk * 0.6 + graph_risk * 0.4
            risk_scores.append(total_risk)
        
        # Select top high-risk nodes
        num_high_risk = int(self.num_nodes * self.high_risk_ratio)
        high_risk_indices = np.argsort(risk_scores)[-num_high_risk:]
        
        labels[high_risk_indices] = 1
        
        return labels
    
    def create_pyg_data(self) -> Data:
        """Create PyTorch Geometric Data object.
        
        Returns:
            Data: PyG Data object with graph, features, and labels.
        """
        # Generate graph
        G = self.generate_graph()
        
        # Convert to edge index
        edge_list = list(G.edges())
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Generate features and labels
        features = self.generate_features(G)
        labels = self.generate_labels(G, features)
        
        # Create train/val/test splits
        train_mask, val_mask, test_mask = self._create_splits()
        
        return Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
    
    def _create_splits(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create train/validation/test splits.
        
        Returns:
            Tuple of boolean masks for train, val, and test sets.
        """
        train_ratio = self.config["train_ratio"]
        val_ratio = self.config["val_ratio"]
        test_ratio = self.config["test_ratio"]
        
        indices = list(range(self.num_nodes))
        train_indices, temp_indices = train_test_split(
            indices, test_size=(1 - train_ratio), random_state=42
        )
        val_indices, test_indices = train_test_split(
            temp_indices, test_size=test_ratio / (val_ratio + test_ratio), random_state=42
        )
        
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return train_mask, val_mask, test_mask


def load_real_world_dataset(dataset_name: str) -> Optional[Data]:
    """Load real-world social good datasets.
    
    Args:
        dataset_name: Name of the dataset to load.
        
    Returns:
        Data: PyG Data object or None if dataset not available.
    """
    # Placeholder for real-world dataset loading
    # In practice, this would load datasets like:
    # - COVID-19 contact networks
    # - Poverty prediction datasets
    # - Disaster response networks
    # - Education access networks
    
    if dataset_name == "covid_contact":
        # Load COVID-19 contact network
        pass
    elif dataset_name == "poverty_prediction":
        # Load poverty prediction dataset
        pass
    else:
        print(f"Dataset {dataset_name} not implemented yet.")
        return None
    
    return None
