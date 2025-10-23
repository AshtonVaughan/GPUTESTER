"""
Utility functions for data processing, normalization, and error handling.
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import torch
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dirs_exist(config: Dict[str, Any]) -> None:
    """Ensure all required directories exist."""
    os.makedirs(config['paths']['data_dir'], exist_ok=True)
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)


def retry_with_backoff(func, max_retries: int = 5, delay: int = 5):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds

    Returns:
        Result of the function call
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def normalize_data(data: pd.DataFrame, columns: list) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize specified columns using StandardScaler.

    Args:
        data: DataFrame containing the data
        columns: List of column names to normalize

    Returns:
        Tuple of (normalized_data, scaler)
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(data[columns].values)
    return normalized, scaler


def create_sequences(data: np.ndarray, seq_length: int, target_col_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.

    Args:
        data: Normalized data array
        seq_length: Length of input sequences
        target_col_idx: Index of the target column

    Returns:
        Tuple of (X, y) where X is input sequences and y is targets
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_col_idx])

    return np.array(X), np.array(y)


def split_data(X: np.ndarray, y: np.ndarray, train_split: float = 0.8, val_split: float = 0.1):
    """
    Split data into train, validation, and test sets.

    Args:
        X: Input sequences
        y: Target values
        train_split: Fraction of data for training
        val_split: Fraction of data for validation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(X)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_device(preferred_device: str = "cuda") -> torch.device:
    """
    Get the appropriate device for training.

    Args:
        preferred_device: Preferred device ('cuda' or 'cpu')

    Returns:
        torch.device object
    """
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def save_training_log(log_path: str, metrics: Dict[str, Any]) -> None:
    """
    Save training metrics to a log file.

    Args:
        log_path: Path to save the log file
        metrics: Dictionary of metrics to save
    """
    import json
    from datetime import datetime

    metrics['timestamp'] = datetime.now().isoformat()

    with open(log_path, 'a') as f:
        f.write(json.dumps(metrics) + '\n')


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def print_model_info(model: torch.nn.Module) -> None:
    """
    Print model architecture and parameter count.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Architecture:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.2f} MB (float32)\n")
