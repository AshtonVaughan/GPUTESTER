# GPUTESTER - Forex Trading Model Trainer

A lightweight, fast-training forex prediction model using PyTorch transformers. Optimized to train in under 10 minutes on modern GPUs (RTX 3090, H100, etc.).

## Features

- **Fast Data Fetching**: Downloads EUR/USD data from Yahoo Finance with retry logic and rate limit handling
- **Lightweight Transformer**: Small but effective transformer architecture optimized for GPU training
- **Production-Ready**: Includes proper error handling, logging, and configuration management
- **Portable**: All paths are relative - clone and run anywhere
- **GPU Optimized**: Designed for fast training on modern NVIDIA GPUs

## Project Structure

```
GPUTESTER/
├── fetch_data.py          # Download EUR/USD data from Yahoo Finance
├── train_model.py         # Train the prediction model
├── model.py               # Model architecture (transformer & LSTM)
├── utils.py               # Utility functions
├── config.yaml            # Configuration settings
├── requirements.txt       # Python dependencies
├── data/                  # Downloaded data storage
├── models/                # Saved model files
├── logs/                  # Training logs
└── README.md             # This file
```

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd GPUTESTER
```

### 2. Install Dependencies

**On your local machine or GPU server:**

```bash
pip install -r requirements.txt
```

**For CUDA support (recommended):**

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Fetch Data

```bash
python fetch_data.py
```

This will:
- Download ~2 years of hourly EUR/USD data from Yahoo Finance
- Handle rate limits with automatic retries
- Save data to `data/eurusd.csv`
- Display data statistics

**Expected output:**
```
Fetching EURUSD=X data from Yahoo Finance
Date range: 2023-XX-XX to 2025-XX-XX
...
Data saved to: ./data/eurusd.csv
Total rows: ~17,520 (2 years of hourly data)
```

### 4. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Create train/validation/test splits
- Train a lightweight transformer model
- Save the best model to `models/model.pt`
- Log training metrics to `logs/`

**Expected training time:**
- RTX 3090: ~5-8 minutes
- H100: ~2-4 minutes
- CPU: ~30-60 minutes

**Expected output:**
```
Training for 50 epochs...
Training: 100%|██████████| 50/50 [05:32<00:00]
Best validation loss: 0.XXXXXX
Test loss: 0.XXXXXX
Model saved to: ./models/model.pt
```

## Configuration

Edit `config.yaml` to customize training:

```yaml
training:
  epochs: 50              # Number of training epochs
  batch_size: 256         # Batch size (reduce if OOM errors)
  learning_rate: 0.001    # Initial learning rate

model:
  hidden_size: 128        # Model capacity
  num_layers: 2           # Transformer layers
  seq_length: 60          # Input sequence length
```

### Common Adjustments

**For slower GPUs or limited VRAM:**
```yaml
training:
  batch_size: 128  # Reduce from 256

model:
  hidden_size: 64  # Reduce from 128
```

**For faster training:**
```yaml
training:
  epochs: 30  # Reduce from 50
```

## GPU Setup Guide

### Check GPU Availability

```bash
# Check if NVIDIA GPU is available
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Troubleshooting GPU Issues

**CUDA Out of Memory:**
```yaml
# In config.yaml, reduce:
training:
  batch_size: 128  # or 64
```

**No GPU detected:**
```yaml
# In config.yaml, fallback to CPU:
device: "cpu"
```

## Data Details

### Source
- **Symbol**: EURUSD=X (Yahoo Finance)
- **Interval**: 1 hour
- **History**: ~2 years (maximum available from Yahoo)
- **Features**: Open, High, Low, Close, Volume

### Preprocessing
1. Normalization using StandardScaler
2. Sequence creation (60 timesteps → 1 prediction)
3. Train/Val/Test split: 80%/10%/10%

## Model Architecture

### Lightweight Transformer
```
Input (5 features) → Linear Projection (128-d)
                   → Positional Encoding
                   → Transformer Encoder (2 layers, 4 heads)
                   → FC Layers (128 → 64 → 1)
                   → Output (price prediction)
```

**Parameters**: ~200K (very lightweight!)

### Alternative: LSTM Model

To use LSTM instead of transformer:

```python
# In train_model.py, change:
model = get_model(config, model_type="lstm")
```

## Training Output

### Files Created

- `models/model.pt`: Best model checkpoint
- `logs/training_YYYYMMDD_HHMMSS.log`: Training metrics (JSON)

### Model Checkpoint Contains

```python
{
    'epoch': int,
    'model_state_dict': state_dict,
    'optimizer_state_dict': state_dict,
    'val_loss': float,
    'config': dict
}
```

### Loading a Trained Model

```python
import torch
from model import get_model
from utils import load_config

# Load config and create model
config = load_config()
model = get_model(config)

# Load checkpoint
checkpoint = torch.load('models/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
# predictions = model(input_tensor)
```

## Error Handling

### Rate Limiting (Yahoo Finance)
The fetcher includes:
- Automatic retry with exponential backoff
- Batch fetching with delays
- Graceful error messages

### Common Issues

**Issue**: `Data file not found`
```bash
Solution: Run python fetch_data.py first
```

**Issue**: `CUDA out of memory`
```yaml
Solution: Reduce batch_size in config.yaml
```

**Issue**: `No data retrieved`
```
Solution: Check internet connection and Yahoo Finance availability
```

## Performance Benchmarks

| GPU         | Batch Size | Training Time |
|-------------|------------|---------------|
| H100        | 256        | ~2-3 min      |
| RTX 3090    | 256        | ~5-8 min      |
| RTX 3080    | 256        | ~7-10 min     |
| CPU (16c)   | 128        | ~30-45 min    |

## Development

### Testing the Model

```bash
# Test model creation
python model.py

# Expected output:
# Model created successfully!
# Total parameters: ~200,000
```

### Adding New Features

1. **Add features to data**: Modify `feature_columns` in `train_model.py`
2. **Change model architecture**: Edit `model.py`
3. **Adjust hyperparameters**: Update `config.yaml`

## Export to GitHub

This project is ready for GitHub:

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: GPUTESTER forex trainer"

# Add remote and push
git remote add origin <your-repo-url>
git push -u origin main
```

### Clone on GPU Server

```bash
# On your GPU server
git clone <your-repo-url>
cd GPUTESTER
pip install -r requirements.txt
python fetch_data.py
python train_model.py
```

## License

MIT License - feel free to use and modify!

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues or questions:
1. Check error messages (they include troubleshooting tips)
2. Review this README
3. Open an issue on GitHub

---

**Built for fast, efficient forex model training on modern GPUs!**
