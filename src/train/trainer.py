"""Training utilities for social good GNN models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from tqdm import tqdm
import wandb

from ..utils.device import get_device, set_seed
from ..utils.config import Config


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """Check if training should stop.
        
        Args:
            val_score: Current validation score.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


class SocialGNNTrainer:
    """Trainer for social good GNN models."""
    
    def __init__(self, config: Config):
        """Initialize trainer.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device()
        set_seed(config.get("seed", 42))
        
        # Initialize logging
        self.use_wandb = config.get("logging.use_wandb", False)
        if self.use_wandb:
            wandb.init(
                project=config.get("logging.wandb_project", "gnn-social-good"),
                config=config.to_dict(),
            )
        
        # Create directories
        self._create_directories()
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_scores = []
        self.test_scores = []
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        os.makedirs(self.config.get("logging.log_dir", "logs"), exist_ok=True)
        os.makedirs(self.config.get("logging.checkpoint_dir", "checkpoints"), exist_ok=True)
    
    def train_epoch(
        self,
        model: nn.Module,
        data: Data,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch.
        
        Args:
            model: Model to train.
            data: Graph data.
            optimizer: Optimizer.
            criterion: Loss function.
            
        Returns:
            Training loss.
        """
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        
        # Gradient clipping
        grad_clip = self.config.get("training.grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        return loss.item()
    
    def evaluate(
        self,
        model: nn.Module,
        data: Data,
        mask: torch.Tensor,
        metrics: List[str],
    ) -> Dict[str, float]:
        """Evaluate model on given mask.
        
        Args:
            model: Model to evaluate.
            data: Graph data.
            mask: Evaluation mask.
            metrics: List of metrics to compute.
            
        Returns:
            Dictionary of metric scores.
        """
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            probs = torch.softmax(out, dim=1)
            
            y_true = data.y[mask]
            y_pred = pred[mask]
            y_probs = probs[mask]
        
        results = {}
        
        # Accuracy
        if "accuracy" in metrics:
            correct = (y_pred == y_true).sum().item()
            total = mask.sum().item()
            results["accuracy"] = correct / total
        
        # F1 scores
        if "f1_macro" in metrics or "f1_micro" in metrics:
            from sklearn.metrics import f1_score
            
            if "f1_macro" in metrics:
                results["f1_macro"] = f1_score(y_true.cpu(), y_pred.cpu(), average="macro")
            
            if "f1_micro" in metrics:
                results["f1_micro"] = f1_score(y_true.cpu(), y_pred.cpu(), average="micro")
        
        # AUROC
        if "auroc" in metrics:
            from sklearn.metrics import roc_auc_score
            
            # For binary classification, use probability of positive class
            if y_probs.shape[1] == 2:
                y_scores = y_probs[:, 1].cpu()
            else:
                y_scores = y_probs.cpu()
            
            results["auroc"] = roc_auc_score(y_true.cpu(), y_scores)
        
        return results
    
    def train(
        self,
        model: nn.Module,
        data: Data,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            model: Model to train.
            data: Graph data.
            optimizer: Optimizer.
            criterion: Loss function.
            scheduler: Learning rate scheduler.
            
        Returns:
            Training history.
        """
        epochs = self.config.get("training.epochs", 100)
        patience = self.config.get("training.patience", 20)
        min_delta = self.config.get("training.min_delta", 1e-4)
        
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
        metrics = self.config.get("evaluation.metrics", ["accuracy", "f1_macro"])
        
        best_val_score = 0.0
        best_model_state = None
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training
            train_loss = self.train_epoch(model, data, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_results = self.evaluate(model, data, data.val_mask, metrics)
            val_score = val_results.get("accuracy", val_results.get("f1_macro", 0.0))
            self.val_scores.append(val_score)
            
            # Test (for monitoring)
            test_results = self.evaluate(model, data, data.test_mask, metrics)
            test_score = test_results.get("accuracy", test_results.get("f1_macro", 0.0))
            self.test_scores.append(test_score)
            
            # Learning rate scheduling
            if scheduler:
                scheduler.step(val_score)
            
            # Logging
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": val_score,
                "test_score": test_score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log_dict.update({f"val_{k}": v for k, v in val_results.items()})
            log_dict.update({f"test_{k}": v for k, v in test_results.items()})
            
            if self.use_wandb:
                wandb.log(log_dict)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, "
                      f"Val Score: {val_score:.4f}, Test Score: {test_score:.4f}")
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = model.state_dict().copy()
                
                # Save checkpoint
                if self.config.get("logging.save_checkpoints", True):
                    self.save_checkpoint(model, optimizer, epoch, val_score)
            
            # Early stopping
            if early_stopping(val_score):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return {
            "train_losses": self.train_losses,
            "val_scores": self.val_scores,
            "test_scores": self.test_scores,
        }
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        score: float,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state.
            epoch: Current epoch.
            score: Current validation score.
        """
        checkpoint_dir = self.config.get("logging.checkpoint_dir", "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}_score_{score:.4f}.pt")
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": score,
            "config": self.config.to_dict(),
        }, checkpoint_path)
    
    def load_checkpoint(self, model: nn.Module, checkpoint_path: str) -> Dict:
        """Load model checkpoint.
        
        Args:
            model: Model to load state into.
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint information.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        return checkpoint
