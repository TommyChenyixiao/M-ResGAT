# M-ResGAT: Multi-hop Motif-based Residual Graph Attention Networks

This repository contains the implementation of M-ResGAT, a novel graph neural network architecture that enhances ResGAT with multi-hop attention mechanisms and motif-based structural information for improved node classification in citation networks.

## Overview

M-ResGAT extends the traditional Graph Attention Network (GAT) by incorporating:
- Multi-hop attention mechanisms to capture broader network context
- Motif-based structural patterns for enhanced graph representation
- Residual connections for better gradient flow
- Hybrid information matrices combining first-order and higher-order relationships

## Model Architecture

The key components of our architecture include:
- Multi-hop attention diffusion layers
- Motif-based hybrid information integration
- Residual connections with layer normalization
- Configurable attention heads and hop distances

## Requirements

```bash
# Core dependencies
torch>=1.9.0
torch-geometric>=2.0.0
torch-scatter>=2.0.9
torch-sparse>=0.6.12
numpy>=1.21.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
PyYAML>=5.1
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/M-ResGAT.git
cd M-ResGAT

# Create a conda environment
conda create -n resgat python=3.8
conda activate resgat

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train on a specific dataset
python train.py --datasets Cora_ML

# Train on multiple datasets
python train.py --datasets Cora_ML CiteSeer

# Train specific models
python train.py --datasets Cora_ML --models GAT ResGAT M-ResGAT

# Use custom configuration
python train.py --datasets Cora_ML --config configs/custom.yaml
```

### Configuration

Modify `config/config.py` to adjust model parameters:
- Model architecture (hidden channels, layers, heads)
- Training parameters (learning rate, weight decay)
- Multi-hop settings (number of hops, combination method)
- Motif attention parameters (beta coefficient)

## Project Structure

```
M-ResGAT/
├── config/
│   ├── __init__.py
│   └── config.py
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── gcn.py
│   ├── graphsage.py
│   ├── gat.py
│   ├── resgat.py
│   ├── mresgat.py
│   └── M_resgat.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logging.py
├── trainers/
│   ├── __init__.py
│   └── trainer.py
├── requirements.txt
└── train.py
```

## Evaluation Metrics

The model's performance is evaluated using:
- Classification Accuracy
- Macro F1-Score
- Node Classification AUC
- Training and Validation Loss Curves
- ROC Curves for Multi-class Classification

## Results


## Citation


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
