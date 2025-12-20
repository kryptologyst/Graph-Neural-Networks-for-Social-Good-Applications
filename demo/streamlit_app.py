"""Streamlit demo for social good GNN applications."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from src.utils.config import Config
from src.utils.device import get_device
from src.data.generator import SocialGraphGenerator
from src.models.social_gnn import create_model
from src.eval.evaluator import SocialGNNEvaluator


def load_model_and_data():
    """Load trained model and data."""
    # Load configuration
    config = Config()
    
    # Generate data
    generator = SocialGraphGenerator(config.get("data", {}))
    data = generator.create_pyg_data()
    
    # Create model
    model_config = config.get("model", {})
    model_config.update({
        "input_dim": data.x.shape[1],
        "output_dim": len(torch.unique(data.y)),
    })
    
    model = create_model(model_config["type"], model_config)
    model = model.to(get_device())
    
    # Load checkpoint if available
    checkpoint_path = "checkpoints/final_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=get_device())
        model.load_state_dict(checkpoint["model_state_dict"])
        st.success("Loaded trained model!")
    else:
        st.warning("No trained model found. Using random weights.")
    
    return model, data, config


def create_interactive_graph(data, predictions=None, probabilities=None):
    """Create interactive graph visualization."""
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(data.num_nodes):
        G.add_node(i)
    
    # Add edges
    edge_list = data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_list)
    
    # Create PyVis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes with attributes
    for node in G.nodes():
        color = "#ff6b6b" if data.y[node] == 1 else "#4ecdc4"  # Red for high risk, teal for low risk
        
        if predictions is not None:
            pred_color = "#ff6b6b" if predictions[node] == 1 else "#4ecdc4"
            color = pred_color
        
        # Node size based on degree
        degree = G.degree(node)
        size = min(20 + degree * 2, 50)
        
        net.add_node(
            node,
            label=f"Node {node}",
            color=color,
            size=size,
            title=f"Node {node}<br>Degree: {degree}<br>True Risk: {'High' if data.y[node] == 1 else 'Low'}"
        )
    
    # Add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
        net.save_graph(tmp_file.name)
        return tmp_file.name


def plot_risk_analysis(data, predictions, probabilities):
    """Create risk analysis plots."""
    risk_scores = probabilities[:, 1].cpu().numpy()
    true_labels = data.y.cpu().numpy()
    pred_labels = predictions.cpu().numpy()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Risk Score Distribution", "Risk by True Label", 
                       "Prediction Confidence", "Risk vs Degree"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Risk score distribution
    fig.add_trace(
        go.Histogram(x=risk_scores, nbinsx=30, name="Risk Scores", opacity=0.7),
        row=1, col=1
    )
    
    # Risk by true label
    high_risk_scores = risk_scores[true_labels == 1]
    low_risk_scores = risk_scores[true_labels == 0]
    
    fig.add_trace(
        go.Histogram(x=low_risk_scores, nbinsx=20, name="Low Risk", opacity=0.7),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=high_risk_scores, nbinsx=20, name="High Risk", opacity=0.7),
        row=1, col=2
    )
    
    # Prediction confidence
    confidence = np.max(probabilities.cpu().numpy(), axis=1)
    fig.add_trace(
        go.Histogram(x=confidence, nbinsx=30, name="Confidence", opacity=0.7),
        row=2, col=1
    )
    
    # Risk vs degree
    degrees = []
    for i in range(data.num_nodes):
        degree = (data.edge_index[0] == i).sum() + (data.edge_index[1] == i).sum()
        degrees.append(degree.item())
    
    fig.add_trace(
        go.Scatter(x=degrees, y=risk_scores, mode='markers', name="Risk vs Degree", opacity=0.6),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Risk Analysis Dashboard")
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GNN Social Good Demo",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Graph Neural Networks for Social Good Applications")
    st.markdown("""
    This demo showcases how Graph Neural Networks can be used for social good applications,
    specifically for predicting social risk levels in contact networks.
    """)
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, data, config = load_model_and_data()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)
        probabilities = torch.softmax(out, dim=1)
    
    # Evaluation metrics
    evaluator = SocialGNNEvaluator(config.get("evaluation", {}))
    results = evaluator._compute_metrics(data.y, predictions, probabilities)
    
    # Display metrics
    st.sidebar.subheader("Model Performance")
    st.sidebar.metric("Accuracy", f"{results['accuracy']:.3f}")
    st.sidebar.metric("F1 Macro", f"{results['f1_macro']:.3f}")
    st.sidebar.metric("AUROC", f"{results['auroc']:.3f}")
    
    # Node exploration
    st.sidebar.subheader("Node Explorer")
    node_id = st.sidebar.selectbox("Select Node", range(min(50, data.num_nodes)))
    
    # Display node information
    st.sidebar.markdown(f"**Node {node_id} Information:**")
    st.sidebar.markdown(f"- True Risk: {'High' if data.y[node_id] == 1 else 'Low'}")
    st.sidebar.markdown(f"- Predicted Risk: {'High' if predictions[node_id] == 1 else 'Low'}")
    st.sidebar.markdown(f"- Risk Score: {probabilities[node_id, 1]:.3f}")
    st.sidebar.markdown(f"- Confidence: {torch.max(probabilities[node_id]):.3f}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Graph Visualization", "Risk Analysis", "Node Features", "Model Insights"])
    
    with tab1:
        st.header("Interactive Graph Visualization")
        st.markdown("""
        The graph below shows the social contact network. Nodes are colored by risk level:
        - 🔴 Red: High risk individuals
        - 🔵 Teal: Low risk individuals
        """)
        
        # Create interactive graph
        graph_file = create_interactive_graph(data, predictions, probabilities)
        
        # Display graph
        with open(graph_file, 'r') as f:
            graph_html = f.read()
        st.components.v1.html(graph_html, height=600)
        
        # Clean up
        os.unlink(graph_file)
    
    with tab2:
        st.header("Risk Analysis Dashboard")
        
        # Risk analysis plots
        risk_fig = plot_risk_analysis(data, predictions, probabilities)
        st.plotly_chart(risk_fig, use_container_width=True)
        
        # Risk statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("High Risk Nodes", f"{data.y.sum().item()}")
        with col2:
            st.metric("Predicted High Risk", f"{predictions.sum().item()}")
        with col3:
            st.metric("Avg Risk Score", f"{probabilities[:, 1].mean():.3f}")
        with col4:
            st.metric("Avg Confidence", f"{torch.max(probabilities, dim=1)[0].mean():.3f}")
    
    with tab3:
        st.header("Node Feature Analysis")
        
        # Feature importance
        feature_names = [
            "Age", "Socioeconomic Status", "Health Condition", "Mobility Level",
            "Community Density", "Geographic Isolation", "Resource Access", "Vulnerability"
        ]
        
        # Calculate feature correlations with risk
        feature_importance = []
        for i in range(data.x.shape[1]):
            feature_values = data.x[:, i].cpu().numpy()
            risk_scores = probabilities[:, 1].cpu().numpy()
            correlation = np.corrcoef(feature_values, risk_scores)[0, 1]
            feature_importance.append(abs(correlation))
        
        # Feature importance plot
        fig = px.bar(
            x=feature_names,
            y=feature_importance,
            title="Feature Importance (Correlation with Risk Prediction)",
            labels={"x": "Features", "y": "Absolute Correlation"}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        feature_cols = st.columns(2)
        for i, feature_name in enumerate(feature_names):
            with feature_cols[i % 2]:
                fig = px.histogram(
                    x=data.x[:, i].cpu().numpy(),
                    title=f"{feature_name} Distribution",
                    nbins=30
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Model Insights")
        
        # Model architecture
        st.subheader("Model Architecture")
        st.code(f"""
        Model Type: {config.get('model.type', 'gcn').upper()}
        Hidden Dimension: {config.get('model.hidden_dim', 64)}
        Number of Layers: {config.get('model.num_layers', 2)}
        Dropout Rate: {config.get('model.dropout', 0.3)}
        """)
        
        # Training configuration
        st.subheader("Training Configuration")
        st.code(f"""
        Learning Rate: {config.get('training.learning_rate', 0.01)}
        Epochs: {config.get('training.epochs', 100)}
        Weight Decay: {config.get('training.weight_decay', 5e-4)}
        """)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", data.num_nodes)
        with col2:
            st.metric("Total Edges", data.num_edges)
        with col3:
            st.metric("Features", data.x.shape[1])
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(data.y.cpu().numpy(), predictions.cpu().numpy())
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Demo
    
    This demo showcases Graph Neural Networks applied to social good problems:
    - **Social Risk Prediction**: Identifying high-risk individuals in contact networks
    - **Pandemic Response**: Modeling disease spread and vulnerability
    - **Disaster Management**: Coordinating emergency response efforts
    - **Resource Allocation**: Optimizing aid distribution in communities
    
    The model uses graph structure and node features to predict risk levels,
    which can help inform public health decisions and resource allocation.
    """)


if __name__ == "__main__":
    main()
