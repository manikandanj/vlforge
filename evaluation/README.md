## Available Evaluators

### 1. PrecisionAtKEvaluator  
Precision@K using binary relevance (species-level matches only)

```yaml
- _target_: "evaluation.precision_at_k_evaluator.PrecisionAtKEvaluator"
  n_samples: 500
  k_values: [1, 5, 15]
  show_progress: true
```

### 2. RecallAtKEvaluator
Recall@K using binary relevance (species-level matches only)

```yaml
- _target_: "evaluation.recall_at_k_evaluator.RecallAtKEvaluator"
  n_samples: 500
  k_values: [1, 5, 15]
  show_progress: true
```

### 3. mAPEvaluator
Mean Average Precision (mAP) using binary relevance

```yaml
- _target_: "evaluation.map_evaluator.mAPEvaluator"
  n_samples: 500
  show_progress: true
```

### 4. NDCGEvaluator
nDCG with hierarchical taxonomic relevance scoring (exponential gain)

Relevance levels: Species=3, Genus=2, Subfamily=1, None=0

```yaml
- _target_: "evaluation.ndcg_evaluator.NDCGEvaluator"
  n_samples: 500
  k_values: [1, 5, 15]
  primary_k: 15  # Primary metric for framework (default: max k)
  show_progress: true
```

## Complete Configuration Example

```yaml
evaluators:
  - _target_: "evaluation.precision_at_k_evaluator.PrecisionAtKEvaluator"
    n_samples: 500
    k_values: [1, 5, 15]
  - _target_: "evaluation.recall_at_k_evaluator.RecallAtKEvaluator"
    n_samples: 500
    k_values: [1, 5, 15]
  - _target_: "evaluation.map_evaluator.mAPEvaluator"
    n_samples: 500
  - _target_: "evaluation.ndcg_evaluator.NDCGEvaluator"
    n_samples: 500
    k_values: [1, 5, 15]
    primary_k: 5
```

