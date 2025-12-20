"""Example usage of the social good GNN project."""

import torch
from src.utils.config import Config
from src.data.generator import SocialGraphGenerator
from src.models.social_gnn import create_model
from src.train.trainer import SocialGNNTrainer
from src.eval.evaluator import SocialGNNEvaluator


def main():
    """Example usage of the social good GNN project."""
    print("🌐 Social Good GNN Example")
    print("=" * 50)
    
    # Load configuration
    config = Config()
    print(f"Configuration loaded: {config.get('data.num_nodes')} nodes")
    
    # Generate synthetic social contact network
    print("\n📊 Generating synthetic social contact network...")
    generator = SocialGraphGenerator(config.get("data", {}))
    data = generator.create_pyg_data()
    
    print(f"Graph created:")
    print(f"  - Nodes: {data.num_nodes}")
    print(f"  - Edges: {data.num_edges}")
    print(f"  - Features: {data.x.shape[1]}")
    print(f"  - Classes: {len(torch.unique(data.y))}")
    print(f"  - High risk ratio: {data.y.sum().item() / data.num_nodes:.2%}")
    
    # Create model
    print("\n🧠 Creating GNN model...")
    model_config = config.get("model", {})
    model_config.update({
        "input_dim": data.x.shape[1],
        "output_dim": len(torch.unique(data.y)),
    })
    
    model = create_model(model_config["type"], model_config)
    print(f"Model created: {model_config['type'].upper()}")
    print(f"  - Hidden dimension: {model_config['hidden_dim']}")
    print(f"  - Number of layers: {model_config['num_layers']}")
    
    # Quick evaluation (without training)
    print("\n📈 Evaluating model (random weights)...")
    evaluator = SocialGNNEvaluator(config.get("evaluation", {}))
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)
    
    results = evaluator._compute_metrics(data.y, pred, probs)
    
    print("Random model performance:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  - {metric}: {value:.4f}")
    
    print("\n✅ Example completed successfully!")
    print("\nTo train the model, run:")
    print("  python scripts/train.py")
    print("\nTo launch the interactive demo, run:")
    print("  streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    main()
