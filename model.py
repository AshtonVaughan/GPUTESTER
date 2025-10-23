"""
Lightweight transformer-based model for forex price prediction.
Optimized for fast training on GPU (under 10 minutes).
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LightweightTransformer(nn.Module):
    """
    Lightweight transformer model for forex prediction.
    Designed to train in under 10 minutes on modern GPUs.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        seq_length: int = 60
    ):
        """
        Args:
            input_size: Number of input features (OHLCV = 5)
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            seq_length: Input sequence length
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=seq_length, dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Take the last timestep
        x = x[:, -1, :]

        # Output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class SimpleLSTM(nn.Module):
    """
    Simple LSTM model as an alternative to transformer.
    Faster to train but potentially less expressive.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_length: int = 60
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last timestep
        last_out = lstm_out[:, -1, :]

        # Fully connected layers
        out = self.fc(last_out)

        return out


def get_model(config: dict, model_type: str = "transformer") -> nn.Module:
    """
    Factory function to create and return a model.

    Args:
        config: Configuration dictionary
        model_type: Type of model ('transformer' or 'lstm')

    Returns:
        PyTorch model
    """
    model_config = config['model']

    if model_type == "transformer":
        model = LightweightTransformer(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            dropout=model_config['dropout'],
            seq_length=model_config['seq_length']
        )
    elif model_type == "lstm":
        model = SimpleLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            dropout=model_config['dropout'],
            seq_length=model_config['seq_length']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Test model creation
    import yaml

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    model = get_model(config, model_type="transformer")
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    seq_length = config['model']['seq_length']
    input_size = config['model']['input_size']

    x = torch.randn(batch_size, seq_length, input_size)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
