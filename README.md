# VLForge

## Overview

VLForge is a modular framework for systematic evaluation of vision models. It addresses common challenges in vision model research: inconsistent evaluation pipelines, difficulty reproducing results across different model/dataset combinations, computational inefficiency of repeated inference, and lack of standardized experiment tracking.

The framework provides:

- **Configurable model/dataset/evaluation swapping** through unified interfaces and configuration files
- **Reproducible evaluation protocols** with controlled randomization and deterministic splits  
- **Fair model comparisons** between base and fine-tuned models on identical experimental setups
- **Comprehensive experiment tracking** with automatic artifact preservation and audit trails
- **Efficient embedding-based workflows** that separate model inference from evaluation

The framework provides a unified interface for model evaluation through configurable components while automatically handling experiment reproducibility, result tracking, and efficient evaluation workflows.

## Architecture

The framework follows a modular design with three core abstractions:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vision Models │    │  Data Loaders   │    │   Evaluators    │
│                 │    │                 │    │                 │
│ • BioCLIP       │    │ • Custom        │    │ • Precision@K   │
│ • OpenAI CLIP   │    │   Datasets      │    │ • Recall@K      │
│ • Fine-tuned    │    │ • Batch         │    │ • mAP           │
│ • Custom Models │    │   Processing    │    │ • nDCG          │
│                 │    │ • Metadata      │    │ • Custom        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │    Configuration        │
                    │      (Hydra)            │
                    └─────────────────────────┘
                                 │
                    ┌─────────────────────────┐
                    │   Execution Modes       │
                    │                         │
                    │ • embed: Generate       │
                    │   embeddings and save   │
                    │                         │
                    │ • eval: Run evaluation  │
                    │   with metrics          │
                    │                         │
                    │ • train: Model training │
                    │   (extensible)          │
                    └─────────────────────────┘
```

## How It Works

### 1. Configuration-Driven Experiments

All experiments are defined through YAML configuration files using Hydra:

```yaml
# Choose your components
model:
  _target_: "models.bioclip.BioCLIPModel"
  model_name: "hf-hub:imageomics/bioclip"

data:
  _target_: "dataloaders.custom_loader.CustomDataLoader"
  base_dir: "/path/to/dataset"
  batch_size: 128

evaluators:
  - _target_: "evaluation.precision_at_k_evaluator.PrecisionAtKEvaluator"
    k_values: [1, 5, 10]
  - _target_: "evaluation.ndcg_evaluator.NDCGEvaluator"
    k_values: [1, 5, 10]
```

### 2. Two-Phase Evaluation Workflow

**Phase 1: Embedding Generation**
```bash
python main.py mode=embed
```
- Processes images through the specified model
- Generates embeddings for the entire dataset
- Saves embeddings and metadata to HDF5 format with auto-generated descriptive filenames
- Creates a permanent record tied to specific model and dataset combinations

**Phase 2: Evaluation**
```bash
python main.py mode=eval evaluation.use_precomputed_embeddings=true
```
- Loads pre-computed embeddings
- Performs query/database splits
- Runs specified evaluators and computes metrics
- Generates comprehensive results with experimental metadata

### 3. Flexible Evaluation Modes

**H5-based Evaluation (Recommended)**
- Uses pre-computed embeddings for fast, consistent evaluation
- Enables multiple evaluation runs without repeated model inference
- Perfect for comparing different metrics or evaluation parameters

**Model-based Evaluation**
- Processes images through models in real-time
- Useful for quick prototyping or when embedding storage isn't needed
- Supports dynamic dataset changes

## Key Features

**Experiment Tracking**: Auto-generated filenames with timestamps and model/dataset identifiers, comprehensive metadata preservation, reproducible data splits with controlled randomization, and structured output directories.

**Plugin Architecture**: Extensible interfaces for vision models (`BaseVisionModel`), data loaders (`BaseDataLoader`), and evaluators (`BaseEvaluator`). Components are composed through Hydra configuration files.

**Evaluation Metrics**: Standard retrieval metrics (Precision@K, Recall@K, mAP) and hierarchical relevance measures (nDCG). Custom evaluators can be implemented by extending the base evaluator class.

## Usage Examples

### Basic Model Comparison
```bash
# Evaluate base model
python main.py model.model_name="hf-hub:imageomics/bioclip" experiment.name="base_model"

# Evaluate fine-tuned model  
python main.py model._target_="models.finetuned_bioclip.FineTunedBioCLIPModel" \
               model.checkpoint_path="/path/to/finetuned.pt" \
               experiment.name="finetuned_model"
```

### Different Evaluation Strategies
```bash
# Quick prototyping with direct inference
python main.py mode=eval evaluation.use_precomputed_embeddings=false

# Production evaluation with pre-computed embeddings
python main.py mode=embed
python main.py mode=eval evaluation.use_precomputed_embeddings=true
```

### Custom Evaluation Parameters
```bash
# Limited evaluation for development
python main.py mode=eval evaluation.split.num_queries=100

# Full evaluation for publication
python main.py mode=eval evaluation.split.num_queries=null
```

## File Structure

```
vlforge/
├── conf/                    # Hydra configuration files
│   ├── experiment.yaml      # Main experiment configuration
│   └── experiment_*.yaml    # Environment-specific configs
├── models/                  # Vision model implementations
│   ├── base.py             # Abstract base class
│   ├── bioclip.py          # BioCLIP implementation
│   └── openai_clip.py      # OpenAI CLIP implementation
├── dataloaders/            # Data loading modules
│   ├── base.py             # Abstract base class
│   └── custom_loader.py    # Dataset-specific loaders
├── evaluation/             # Evaluation metrics
│   ├── base.py             # Abstract base class
│   ├── precision_at_k_evaluator.py
│   ├── recall_at_k_evaluator.py
│   ├── map_evaluator.py
│   └── ndcg_evaluator.py
├── embeddings/             # Embedding storage utilities
├── utils/                  # Utility functions
└── main.py                 # Main entry point
```

## Getting Started

1. **Configure your experiment** in `conf/experiment.yaml`
2. **Generate embeddings**: `python main.py mode=embed`
3. **Run evaluation**: `python main.py mode=eval`
4. **Compare results** across different configurations

The framework automatically handles experiment tracking, file organization, and result documentation, allowing you to focus on the research questions rather than implementation details.
