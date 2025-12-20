"""Evaluation utilities for social good GNN models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..utils.device import get_device


class SocialGNNEvaluator:
    """Evaluator for social good GNN models."""
    
    def __init__(self, config: dict):
        """Initialize evaluator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        self.results = {}
    
    def comprehensive_evaluation(
        self,
        model: torch.nn.Module,
        data,
        save_results: bool = True,
    ) -> Dict[str, float]:
        """Perform comprehensive evaluation.
        
        Args:
            model: Trained model.
            data: Graph data.
            save_results: Whether to save results to files.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            probs = torch.softmax(out, dim=1)
        
        # Evaluate on different splits
        splits = {
            "train": data.train_mask,
            "val": data.val_mask,
            "test": data.test_mask,
        }
        
        all_results = {}
        
        for split_name, mask in splits.items():
            if mask.sum() == 0:
                continue
                
            y_true = data.y[mask].cpu()
            y_pred = pred[mask].cpu()
            y_probs = probs[mask].cpu()
            
            results = self._compute_metrics(y_true, y_pred, y_probs)
            all_results[split_name] = results
        
        # Overall results
        overall_results = self._compute_metrics(
            data.y.cpu(), pred.cpu(), probs.cpu()
        )
        all_results["overall"] = overall_results
        
        # Additional analysis
        if save_results:
            self._save_predictions(data, pred, probs)
            self._create_visualizations(data, pred, probs)
            self._analyze_embeddings(model, data)
        
        self.results = all_results
        return all_results
    
    def _compute_metrics(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        y_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_probs: Prediction probabilities.
            
        Returns:
            Dictionary of metrics.
        """
        y_true_np = y_true.numpy()
        y_pred_np = y_pred.numpy()
        
        # Basic metrics
        accuracy = accuracy_score(y_true_np, y_pred_np)
        f1_macro = f1_score(y_true_np, y_pred_np, average="macro")
        f1_micro = f1_score(y_true_np, y_pred_np, average="micro")
        
        # AUROC
        if y_probs.shape[1] == 2:
            auroc = roc_auc_score(y_true_np, y_probs[:, 1].numpy())
        else:
            auroc = roc_auc_score(y_true_np, y_probs.numpy(), multi_class="ovr")
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(
            y_true_np, y_probs[:, 1].numpy() if y_probs.shape[1] == 2 else y_probs.numpy()
        )
        pr_auc = np.trapz(precision, recall)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_np, y_pred_np)
        
        # Additional metrics for imbalanced data
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "auroc": auroc,
            "pr_auc": pr_auc,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "confusion_matrix": cm,
        }
    
    def _save_predictions(
        self,
        data,
        pred: torch.Tensor,
        probs: torch.Tensor,
    ) -> None:
        """Save predictions to file.
        
        Args:
            data: Graph data.
            pred: Predictions.
            probs: Probabilities.
        """
        if not self.config.get("evaluation.save_predictions", True):
            return
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            "node_id": range(len(pred)),
            "true_label": data.y.cpu().numpy(),
            "predicted_label": pred.cpu().numpy(),
            "prob_class_0": probs[:, 0].cpu().numpy(),
            "prob_class_1": probs[:, 1].cpu().numpy(),
            "train_mask": data.train_mask.cpu().numpy(),
            "val_mask": data.val_mask.cpu().numpy(),
            "test_mask": data.test_mask.cpu().numpy(),
        })
        
        # Add node features
        feature_names = [f"feature_{i}" for i in range(data.x.shape[1])]
        for i, name in enumerate(feature_names):
            results_df[name] = data.x[:, i].cpu().numpy()
        
        # Save to CSV
        os.makedirs("assets", exist_ok=True)
        results_df.to_csv("assets/predictions.csv", index=False)
    
    def _create_visualizations(
        self,
        data,
        pred: torch.Tensor,
        probs: torch.Tensor,
    ) -> None:
        """Create visualization plots.
        
        Args:
            data: Graph data.
            pred: Predictions.
            probs: Probabilities.
        """
        os.makedirs("assets/plots", exist_ok=True)
        
        # Confusion matrix
        self._plot_confusion_matrix(data, pred)
        
        # ROC curve
        self._plot_roc_curve(data, probs)
        
        # Feature importance
        self._plot_feature_importance(data, probs)
        
        # Risk distribution
        self._plot_risk_distribution(data, pred, probs)
    
    def _plot_confusion_matrix(self, data, pred: torch.Tensor) -> None:
        """Plot confusion matrix."""
        y_true = data.y.cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("assets/plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_roc_curve(self, data, probs: torch.Tensor) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve
        
        y_true = data.y.cpu().numpy()
        y_scores = probs[:, 1].cpu().numpy()
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auroc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("assets/plots/roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_feature_importance(self, data, probs: torch.Tensor) -> None:
        """Plot feature importance analysis."""
        # Simple feature importance based on correlation with predictions
        feature_importance = []
        
        for i in range(data.x.shape[1]):
            feature_values = data.x[:, i].cpu().numpy()
            pred_scores = probs[:, 1].cpu().numpy()
            correlation = np.corrcoef(feature_values, pred_scores)[0, 1]
            feature_importance.append(abs(correlation))
        
        feature_names = [f"Feature {i}" for i in range(data.x.shape[1])]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_names, feature_importance)
        plt.title("Feature Importance (Correlation with Risk Prediction)")
        plt.xlabel("Features")
        plt.ylabel("Absolute Correlation")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("assets/plots/feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _plot_risk_distribution(
        self,
        data,
        pred: torch.Tensor,
        probs: torch.Tensor,
    ) -> None:
        """Plot risk distribution analysis."""
        risk_scores = probs[:, 1].cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Risk score distribution
        axes[0, 0].hist(risk_scores, bins=30, alpha=0.7, color="skyblue")
        axes[0, 0].set_title("Risk Score Distribution")
        axes[0, 0].set_xlabel("Risk Score")
        axes[0, 0].set_ylabel("Frequency")
        
        # Risk by true labels
        high_risk_scores = risk_scores[data.y.cpu().numpy() == 1]
        low_risk_scores = risk_scores[data.y.cpu().numpy() == 0]
        
        axes[0, 1].hist([low_risk_scores, high_risk_scores], bins=20, 
                       alpha=0.7, label=["Low Risk", "High Risk"], color=["green", "red"])
        axes[0, 1].set_title("Risk Score by True Label")
        axes[0, 1].set_xlabel("Risk Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        
        # Risk by degree (if available)
        degrees = torch.zeros(data.x.shape[0])
        for i in range(data.x.shape[0]):
            degrees[i] = (data.edge_index[0] == i).sum() + (data.edge_index[1] == i).sum()
        
        axes[1, 0].scatter(degrees.numpy(), risk_scores, alpha=0.6)
        axes[1, 0].set_title("Risk Score vs Node Degree")
        axes[1, 0].set_xlabel("Node Degree")
        axes[1, 0].set_ylabel("Risk Score")
        
        # Prediction confidence
        confidence = torch.max(probs, dim=1)[0].cpu().numpy()
        axes[1, 1].hist(confidence, bins=30, alpha=0.7, color="orange")
        axes[1, 1].set_title("Prediction Confidence Distribution")
        axes[1, 1].set_xlabel("Confidence")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig("assets/plots/risk_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def _analyze_embeddings(self, model: torch.nn.Module, data) -> None:
        """Analyze learned embeddings."""
        if not self.config.get("evaluation.save_embeddings", True):
            return
        
        model.eval()
        
        with torch.no_grad():
            # Get embeddings from the last hidden layer
            x = data.x
            for i, layer in enumerate(model.layers[:-1]):  # All layers except the last
                if hasattr(layer, 'conv'):
                    x = layer.conv(x, data.edge_index)
                else:
                    x = layer(x, data.edge_index)
                x = torch.relu(x)
            
            embeddings = x.cpu().numpy()
        
        # t-SNE visualization
        if embeddings.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
        else:
            embeddings_2d = embeddings
        
        # Create interactive plot
        fig = go.Figure()
        
        # Color by true labels
        colors = data.y.cpu().numpy()
        
        fig.add_trace(go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(title="Risk Level")
            ),
            text=[f"Node {i}" for i in range(len(embeddings_2d))],
            hovertemplate="Node: %{text}<br>Risk Level: %{marker.color}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Node Embeddings Visualization (t-SNE)",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            width=800,
            height=600
        )
        
        fig.write_html("assets/plots/embeddings_tsne.html")
        
        # Save embeddings
        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.to_csv("assets/embeddings/node_embeddings.csv", index=False)
    
    def create_model_comparison(self, results_dict: Dict[str, Dict]) -> None:
        """Create model comparison visualization.
        
        Args:
            results_dict: Dictionary of results from different models.
        """
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, results in results_dict.items():
            for split, metrics in results.items():
                if isinstance(metrics, dict) and "accuracy" in metrics:
                    row = {"Model": model_name, "Split": split}
                    row.update(metrics)
                    comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        metrics_to_plot = ["accuracy", "f1_macro", "auroc"]
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in df.columns:
                pivot_df = df.pivot(index="Model", columns="Split", values=metric)
                pivot_df.plot(kind="bar", ax=axes[i], alpha=0.8)
                axes[i].set_title(f"{metric.upper()} Comparison")
                axes[i].set_ylabel(metric.upper())
                axes[i].legend(title="Split")
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("assets/plots/model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Save comparison table
        df.to_csv("assets/model_comparison.csv", index=False)
    
    def generate_report(self) -> str:
        """Generate evaluation report.
        
        Returns:
            Formatted evaluation report.
        """
        if not self.results:
            return "No evaluation results available."
        
        report = "# Social Good GNN Evaluation Report\n\n"
        
        for split, metrics in self.results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                report += f"## {split.upper()} Results\n\n"
                
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
                
                report += "\n"
        
        return report
