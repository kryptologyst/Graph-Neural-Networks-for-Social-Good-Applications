"""Configuration management using OmegaConf."""

from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the GNN social good project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path:
            self.config = OmegaConf.load(config_path)
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration.
        
        Returns:
            DictConfig: Default configuration.
        """
        default_config = {
            # Data configuration
            "data": {
                "name": "synthetic_social",
                "num_nodes": 1000,
                "num_features": 8,
                "num_classes": 2,
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "high_risk_ratio": 0.2,
                "graph_type": "barabasi_albert",
                "m": 3,  # Barabasi-Albert parameter
            },
            
            # Model configuration
            "model": {
                "type": "gcn",
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "activation": "relu",
                "normalization": "batch",
                "residual": True,
            },
            
            # Training configuration
            "training": {
                "epochs": 100,
                "batch_size": 1,  # Full graph training
                "learning_rate": 0.01,
                "weight_decay": 5e-4,
                "patience": 20,
                "min_delta": 1e-4,
                "grad_clip": 1.0,
            },
            
            # Evaluation configuration
            "evaluation": {
                "metrics": ["accuracy", "f1_macro", "f1_micro", "auroc"],
                "save_predictions": True,
                "save_embeddings": True,
            },
            
            # Logging configuration
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "use_wandb": False,
                "wandb_project": "gnn-social-good",
                "save_checkpoints": True,
                "checkpoint_dir": "checkpoints",
            },
            
            # Device configuration
            "device": {
                "auto": True,
                "device_name": "auto",
            },
            
            # Reproducibility
            "seed": 42,
        }
        
        return OmegaConf.create(default_config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.
            
        Returns:
            Configuration value.
        """
        return OmegaConf.select(self.config, key, default=default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.
        
        Args:
            updates: Dictionary of updates.
        """
        OmegaConf.set(self.config, updates)
    
    def save(self, path: str) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration.
        """
        OmegaConf.save(self.config, path)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return OmegaConf.to_container(self.config, resolve=True)
