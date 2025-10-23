"""
Train the forex prediction model.
Optimized for fast training on modern GPUs (under 10 minutes).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from model import get_model
from utils import (
    load_config,
    ensure_dirs_exist,
    normalize_data,
    create_sequences,
    split_data,
    get_device,
    save_training_log,
    EarlyStopping,
    print_model_info
)


def load_and_preprocess_data(config: dict):
    """
    Load and preprocess the forex data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    print("\n" + "="*60)
    print("Loading and preprocessing data...")
    print("="*60 + "\n")

    # Load data
    data_path = config['paths']['data_file']
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Please run 'python fetch_data.py' first to download the data."
        )

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(df):,} rows from {data_path}")

    # Use OHLCV columns
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Normalize data
    normalized_data, scaler = normalize_data(df, feature_columns)
    print(f"Normalized features: {feature_columns}")

    # Create sequences
    seq_length = config['model']['seq_length']
    target_col_idx = feature_columns.index(config['data']['target_column'])
    X, y = create_sequences(normalized_data, seq_length, target_col_idx)
    print(f"Created {len(X):,} sequences with length {seq_length}")

    # Split data
    train_split = config['data']['train_split']
    val_split = config['data']['val_split']
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_split, val_split
    )

    print(f"\nData splits:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test:       {len(X_test):,} samples")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # Create data loaders
    batch_size = config['training']['batch_size']

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"\nBatch size: {batch_size}")
    print(f"Training batches: {len(train_loader)}")

    return train_loader, val_loader, test_loader, scaler


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_model(config: dict):
    """
    Main training function.

    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("Starting model training")
    print("="*60 + "\n")

    start_time = time.time()

    # Get device
    device = get_device(config['device'])

    # Load and preprocess data
    train_loader, val_loader, test_loader, scaler = load_and_preprocess_data(config)

    # Create model
    print("\nCreating model...")
    model = get_model(config, model_type="transformer")
    model = model.to(device)
    print_model_info(model)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=1e-6)

    # Training loop
    epochs = config['training']['epochs']
    gradient_clip = config['training']['gradient_clip']
    best_val_loss = float('inf')
    training_history = []

    print(f"Training for {epochs} epochs...")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Gradient clipping: {gradient_clip}\n")

    progress_bar = tqdm(range(epochs), desc="Training")

    for epoch in progress_bar:
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"\nLearning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, config['paths']['model_file'])

        # Update progress bar
        progress_bar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'best_val': f'{best_val_loss:.6f}'
        })

        # Save to history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    # Test the model
    print("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, criterion, device)

    # Calculate training time
    training_time = time.time() - start_time
    minutes, seconds = divmod(training_time, 60)

    # Print final results
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Training time: {int(minutes)}m {int(seconds)}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Model saved to: {config['paths']['model_file']}")
    print("="*60 + "\n")

    # Save training log
    log_file = os.path.join(
        config['paths']['logs_dir'],
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    save_training_log(log_file, {
        'training_time_seconds': training_time,
        'epochs_trained': len(training_history),
        'best_val_loss': best_val_loss,
        'test_loss': test_loss,
        'history': training_history
    })
    print(f"Training log saved to: {log_file}\n")


def main():
    """Main function."""
    try:
        # Load configuration
        config = load_config()

        # Ensure directories exist
        ensure_dirs_exist(config)

        # Train model
        train_model(config)

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease run 'python fetch_data.py' first to download the data.\n")
        raise

    except RuntimeError as e:
        if "CUDA" in str(e) or "device" in str(e).lower():
            print(f"\nGPU Error: {str(e)}")
            print("\nTroubleshooting tips:")
            print("1. Check if CUDA is properly installed: nvidia-smi")
            print("2. Verify PyTorch CUDA compatibility: python -c 'import torch; print(torch.cuda.is_available())'")
            print("3. Try reducing batch_size in config.yaml")
            print("4. Set device to 'cpu' in config.yaml if GPU is unavailable\n")
        raise

    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("\nDebug information:")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
