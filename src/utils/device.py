"""Utility functions for device management and deterministic seeding."""

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device with fallback chain: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For MPS (Apple Silicon)
    if hasattr(torch.backends, "mps"):
        os.environ["PYTHONHASHSEED"] = str(seed)


def get_device_info() -> dict:
    """Get information about the current device.
    
    Returns:
        dict: Device information including name, memory, and capabilities.
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_reserved": torch.cuda.memory_reserved(device),
            "capability": torch.cuda.get_device_capability(device),
        })
    elif device.type == "mps":
        info.update({
            "name": "Apple Silicon GPU (MPS)",
            "available": torch.backends.mps.is_available(),
        })
    else:
        info.update({
            "name": "CPU",
            "threads": torch.get_num_threads(),
        })
    
    return info
