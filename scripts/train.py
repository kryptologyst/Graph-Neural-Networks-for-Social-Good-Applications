"""Main training script for social good GNN models."""

import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

from src.utils.config import Config
from src.utils.device import get_device, set_seed, get_device_info
from src.data.generator import SocialGraphGenerator
from src.models.social_gnn import create_model
from src.train.trainer import SocialGNNTrainer
from src.eval.evaluator import SocialGNNEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GNN for social good applications")
    
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage", "gat"], help="Model type")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    
    return parser.parse_args()


def create_data(config: Config) -> Data:
    """Create or load dataset.
    
    Args:
        config: Configuration object.
        
    Returns:
        PyG Data object.
    """
    data_config = config.get("data", {})
    
    if data_config.get("name") == "synthetic_social":
        generator = SocialGraphGenerator(data_config)
        data = generator.create_pyg_data()
    else:
        # Load real-world dataset
        from src.data.generator import load_real_world_dataset
        data = load_real_world_dataset(data_config.get("name"))
        
        if data is None:
            print("Falling back to synthetic data...")
            generator = SocialGraphGenerator(data_config)
            data = generator.create_pyg_data()
    
    return data


def create_model_and_optimizer(config: Config, data: Data) -> tuple:
    """Create model and optimizer.
    
    Args:
        config: Configuration object.
        data: Graph data.
        
    Returns:
        Tuple of (model, optimizer, criterion).
    """
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    
    # Model configuration
    model_config.update({
        "input_dim": data.x.shape[1],
        "output_dim": len(torch.unique(data.y)),
    })
    
    # Create model
    model = create_model(model_config["type"], model_config)
    model = model.to(get_device())
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get("learning_rate", 0.01),
        weight_decay=training_config.get("weight_decay", 5e-4),
    )
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def train_model(config: Config, data: Data) -> Dict:
    """Train the model.
    
    Args:
        config: Configuration object.
        data: Graph data.
        
    Returns:
        Training results.
    """
    # Create model and optimizer
    model, optimizer, criterion = create_model_and_optimizer(config, data)
    
    # Move data to device
    data = data.to(get_device())
    
    # Create trainer
    trainer = SocialGNNTrainer(config)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )
    
    # Train
    history = trainer.train(model, data, optimizer, criterion, scheduler)
    
    return {
        "model": model,
        "history": history,
        "trainer": trainer,
    }


def evaluate_model(config: Config, data: Data, model: nn.Module) -> Dict:
    """Evaluate the model.
    
    Args:
        config: Configuration object.
        data: Graph data.
        model: Trained model.
        
    Returns:
        Evaluation results.
    """
    data = data.to(get_device())
    
    evaluator = SocialGNNEvaluator(config.get("evaluation", {}))
    results = evaluator.comprehensive_evaluation(model, data)
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = Config(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.model:
        config.update({"model": {"type": args.model}})
    if args.epochs:
        config.update({"training": {"epochs": args.epochs}})
    if args.lr:
        config.update({"training": {"learning_rate": args.lr}})
    if args.hidden_dim:
        config.update({"model": {"hidden_dim": args.hidden_dim}})
    if args.num_layers:
        config.update({"model": {"num_layers": args.num_layers}})
    if args.dropout:
        config.update({"model": {"dropout": args.dropout}})
    if args.seed:
        config.update({"seed": args.seed})
    if args.wandb:
        config.update({"logging": {"use_wandb": True}})
    
    # Set seed
    set_seed(config.get("seed", 42))
    
    # Print device info
    device_info = get_device_info()
    print(f"Using device: {device_info}")
    
    # Create data
    print("Creating dataset...")
    data = create_data(config)
    print(f"Dataset created: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Features: {data.x.shape[1]}, Classes: {len(torch.unique(data.y))}")
    
    if args.eval_only:
        # Evaluation only
        if args.checkpoint:
            model, _, _ = create_model_and_optimizer(config, data)
            checkpoint = torch.load(args.checkpoint, map_location=get_device())
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {args.checkpoint}")
        else:
            print("Error: --checkpoint required for evaluation only mode")
            return
        
        results = evaluate_model(config, data, model)
        print("Evaluation Results:")
        for split, metrics in results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                print(f"\n{split.upper()}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
    else:
        # Training
        print("Starting training...")
        training_results = train_model(config, data)
        model = training_results["model"]
        
        # Evaluation
        print("Evaluating model...")
        eval_results = evaluate_model(config, data, model)
        
        # Print results
        print("\nFinal Results:")
        for split, metrics in eval_results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                print(f"\n{split.upper()}:")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
        
        # Save final model
        checkpoint_path = "checkpoints/final_model.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "results": eval_results,
        }, checkpoint_path)
        print(f"\nModel saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
