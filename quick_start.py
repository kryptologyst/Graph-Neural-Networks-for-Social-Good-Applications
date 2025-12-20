#!/usr/bin/env python3
"""Quick start script for the Social Good GNN project."""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🌐 Graph Neural Networks for Social Good Applications")
    print("=" * 60)
    print("This script will help you get started with the project.")
    print("It will install dependencies, run tests, and demonstrate the system.")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required. Current version:", sys.version)
        return
    
    print(f"✅ Python version: {sys.version}")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies. Please check requirements.txt")
        return
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    if not run_command("python scripts/example.py", "Running example"):
        print("❌ Failed to run example")
        return
    
    # Check if Streamlit is available
    try:
        import streamlit
        print("\n🎉 Installation completed successfully!")
        print("\nNext steps:")
        print("1. Train a model:")
        print("   python scripts/train.py")
        print("\n2. Launch interactive demo:")
        print("   streamlit run demo/streamlit_app.py")
        print("\n3. Explore the code:")
        print("   - src/models/ - GNN implementations")
        print("   - src/data/ - Data generation")
        print("   - src/eval/ - Evaluation metrics")
        print("   - configs/ - Configuration files")
        
    except ImportError:
        print("\n⚠️  Streamlit not available. Install with:")
        print("   pip install streamlit")
        print("\nThen run: streamlit run demo/streamlit_app.py")


if __name__ == "__main__":
    main()
