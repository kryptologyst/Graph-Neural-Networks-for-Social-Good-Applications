"""Test suite for social good GNN project."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from src.utils.device import get_device, set_seed
from src.utils.config import Config
from src.data.generator import SocialGraphGenerator
from src.models.social_gnn import create_model, SocialRiskGCN, SocialRiskSAGE, SocialRiskGAT
from src.train.trainer import SocialGNNTrainer, EarlyStopping
from src.eval.evaluator import SocialGNNEvaluator


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are deterministic
        torch.manual_seed(42)
        rand1 = torch.rand(5)
        torch.manual_seed(42)
        rand2 = torch.rand(5)
        assert torch.allclose(rand1, rand2)


class TestConfig:
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.get("seed") == 42
        assert config.get("data.num_nodes") == 1000
        assert config.get("model.type") == "gcn"
    
    def test_config_update(self):
        """Test configuration updates."""
        config = Config()
        config.update({"seed": 123})
        assert config.get("seed") == 123


class TestDataGenerator:
    """Test data generation."""
    
    def test_social_graph_generator(self):
        """Test social graph generation."""
        config = {
            "num_nodes": 100,
            "num_features": 4,
            "num_classes": 2,
            "high_risk_ratio": 0.2,
            "graph_type": "barabasi_albert",
            "m": 2
        }
        
        generator = SocialGraphGenerator(config)
        data = generator.create_pyg_data()
        
        assert isinstance(data, Data)
        assert data.num_nodes == 100
        assert data.x.shape[1] == 4
        assert len(torch.unique(data.y)) == 2
        assert data.train_mask.sum() > 0
        assert data.val_mask.sum() > 0
        assert data.test_mask.sum() > 0
    
    def test_feature_generation(self):
        """Test feature generation."""
        import networkx as nx
        
        G = nx.barabasi_albert_graph(50, 2)
        config = {
            "num_nodes": 50,
            "num_features": 6,
            "num_classes": 2,
            "high_risk_ratio": 0.2,
            "graph_type": "barabasi_albert",
            "m": 2
        }
        
        generator = SocialGraphGenerator(config)
        features = generator.generate_features(G)
        
        assert features.shape == (50, 6)
        assert torch.all(features >= 0)  # Features should be non-negative
        assert torch.all(features <= 1)  # Features should be normalized


class TestModels:
    """Test model implementations."""
    
    def test_gcn_model(self):
        """Test GCN model."""
        model = SocialRiskGCN(
            input_dim=8,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            dropout=0.1
        )
        
        x = torch.randn(100, 8)
        edge_index = torch.randint(0, 100, (2, 200))
        
        out = model(x, edge_index)
        assert out.shape == (100, 2)
    
    def test_sage_model(self):
        """Test GraphSAGE model."""
        model = SocialRiskSAGE(
            input_dim=8,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            dropout=0.1
        )
        
        x = torch.randn(100, 8)
        edge_index = torch.randint(0, 100, (2, 200))
        
        out = model(x, edge_index)
        assert out.shape == (100, 2)
    
    def test_gat_model(self):
        """Test GAT model."""
        model = SocialRiskGAT(
            input_dim=8,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        x = torch.randn(100, 8)
        edge_index = torch.randint(0, 100, (2, 200))
        
        out = model(x, edge_index)
        assert out.shape == (100, 2)
        
        # Test attention weights
        attention_weights = model.get_attention_weights(x, edge_index)
        assert len(attention_weights) == 2  # Two layers
    
    def test_create_model(self):
        """Test model creation function."""
        config = {
            "input_dim": 8,
            "hidden_dim": 32,
            "output_dim": 2,
            "num_layers": 2,
            "dropout": 0.1
        }
        
        for model_type in ["gcn", "sage", "gat"]:
            model = create_model(model_type, config)
            assert isinstance(model, torch.nn.Module)


class TestTrainer:
    """Test training utilities."""
    
    def test_early_stopping(self):
        """Test early stopping."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)
        
        # Test improvement
        assert not early_stop(0.8)
        assert not early_stop(0.9)
        
        # Test no improvement
        assert not early_stop(0.89)  # Below threshold
        assert not early_stop(0.88)
        assert not early_stop(0.87)
        assert early_stop(0.86)  # Should trigger early stopping
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = Config()
        trainer = SocialGNNTrainer(config)
        assert trainer.config is not None
        assert trainer.device is not None


class TestEvaluator:
    """Test evaluation utilities."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {"save_predictions": True, "save_embeddings": True}
        evaluator = SocialGNNEvaluator(config)
        assert evaluator.config == config
    
    def test_metrics_computation(self):
        """Test metrics computation."""
        config = {}
        evaluator = SocialGNNEvaluator(config)
        
        y_true = torch.tensor([0, 1, 0, 1, 0])
        y_pred = torch.tensor([0, 1, 0, 0, 0])
        y_probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.7, 0.3]])
        
        metrics = evaluator._compute_metrics(y_true, y_pred, y_probs)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "auroc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert 0 <= metrics["auroc"] <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create small dataset
        config = {
            "num_nodes": 50,
            "num_features": 4,
            "num_classes": 2,
            "high_risk_ratio": 0.2,
            "graph_type": "barabasi_albert",
            "m": 2
        }
        
        generator = SocialGraphGenerator(config)
        data = generator.create_pyg_data()
        
        # Create model
        model_config = {
            "type": "gcn",
            "input_dim": 4,
            "hidden_dim": 16,
            "output_dim": 2,
            "num_layers": 2,
            "dropout": 0.1
        }
        
        model = create_model("gcn", model_config)
        
        # Test forward pass
        out = model(data.x, data.edge_index)
        assert out.shape == (50, 2)
        
        # Test loss computation
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        assert loss.item() > 0
    
    def test_model_comparison(self):
        """Test model comparison."""
        config = {
            "num_nodes": 50,
            "num_features": 4,
            "num_classes": 2,
            "high_risk_ratio": 0.2,
            "graph_type": "barabasi_albert",
            "m": 2
        }
        
        generator = SocialGraphGenerator(config)
        data = generator.create_pyg_data()
        
        model_config = {
            "input_dim": 4,
            "hidden_dim": 16,
            "output_dim": 2,
            "num_layers": 2,
            "dropout": 0.1
        }
        
        results = {}
        
        for model_type in ["gcn", "sage", "gat"]:
            model = create_model(model_type, model_config)
            model.eval()
            
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                probs = torch.softmax(out, dim=1)
            
            evaluator = SocialGNNEvaluator({})
            metrics = evaluator._compute_metrics(data.y, pred, probs)
            results[model_type] = {"test": metrics}
        
        # All models should produce valid results
        assert len(results) == 3
        for model_name, result in results.items():
            assert "test" in result
            assert "accuracy" in result["test"]


if __name__ == "__main__":
    pytest.main([__file__])
