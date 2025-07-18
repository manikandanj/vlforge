# VLForge

## Overview

VLForge is a vision-language model evaluation framework designed for biological image classification tasks.

## Usage

### Modes of Operation

The framework supports three main modes:

#### 1. Embedding Generation (`mode: "embed"`)
Generates embeddings using a model and dataloader, saves results to HDF5 format:
```bash
python main.py mode=embed
```

#### 2. Evaluation (`mode: "eval"`)
Runs evaluation with two available approaches:

**H5-based Evaluation (Default - Fast)**
```bash
python main.py mode=eval evaluation.use_precomputed_embeddings=true
```
- Uses pre-computed embeddings from HDF5 files
- Much faster as no model inference is required
- Ideal for repeated evaluations with same embeddings

**Model-based Evaluation (Flexible)**
```bash
python main.py mode=eval evaluation.use_precomputed_embeddings=false
```
- Processes images through model in real-time
- Slower but more flexible (can evaluate different datasets without pre-computing)
- Useful for testing model changes or evaluating on new datasets

#### 3. Training (`mode: "train"`)
Training functionality (not yet implemented).

### Configuration

Key configuration options in `conf/experiment.yaml`:

```yaml
# Choose operation mode
mode: "eval"  # "embed" | "eval" | "train"

# Evaluation configuration
evaluation:
  # Choose evaluation approach
  use_precomputed_embeddings: true  # true = H5-based, false = model-based
  
  # H5-based settings (when use_precomputed_embeddings: true)
  embedding_file: null  # Specific file or null for most recent
  split:
    query_ratio: 0.01
    num_queries: 10
```

### Examples

1. **Generate embeddings then evaluate (recommended workflow):**
```bash
# Step 1: Generate embeddings
python main.py mode=embed

# Step 2: Evaluate using H5 file (fast)
python main.py mode=eval evaluation.use_precomputed_embeddings=true
```

2. **Direct model evaluation (for testing/prototyping):**
```bash
python main.py mode=eval evaluation.use_precomputed_embeddings=false
```

## File Structure

```
vlforge/
├── conf/                    # Hydra configuration files
├── dataloaders/            # Data loading modules  
├── embeddings/             # Embedding storage utilities
├── evaluation/             # Evaluation metrics
├── models/                 # Model implementations
├── utils/                  # Utility functions
└── main.py                 # Main entry point
```
