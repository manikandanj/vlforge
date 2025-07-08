from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .base import BaseEvaluator
from PIL import Image
import numpy as np
import pandas as pd


class PrecisionAtKEvaluator(BaseEvaluator):
    """
    Evaluator that calculates Precision@K using binary relevance (species-level).
    Only exact species matches are considered relevant.
    """
    
    def __init__(self, n_samples: int = 200, k_values: List[int] = None, 
                 show_progress: bool = True, **kwargs):
        self.k_values = k_values or [1, 5, 10, 15]
        metric_names = [f"precision@{k}" for k in self.k_values]
        super().__init__(n_samples=n_samples, metrics=metric_names, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress
        
    def precision_at_k(self, query_species: str, ranked_species: List[str], k: int) -> float:
        """
        Calculate Precision@k using binary relevance (species-level).
        
        Args:
            query_species: True species of the query image
            ranked_species: List of species in ranked order of similarity
            k: Number of top results to consider
        
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k <= 0 or len(ranked_species) == 0:
            return 0.0
        
        top_k_species = ranked_species[:k]
        relevant_count = sum(1 for species in top_k_species if species == query_species)
        precision = relevant_count / k
        
        return precision
        
    def get_ranked_predictions(self, model, query_images: List[Image.Image], 
                             candidate_labels: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Get ranked predictions from the model for query images.
        Returns list of lists: each inner list contains (predicted_label, confidence_score) 
        tuples sorted by confidence for one query image.
        """
        predictions = []
        
        for query_image in query_images:
            # Get similarities for this query image against all candidate labels
            similarities = model.compute_similarity([query_image], candidate_labels)
            
            # Convert to probabilities and get ranked results
            probs = similarities.softmax(dim=-1)[0]  # Take first (and only) row
            
            # Sort by probability descending
            sorted_indices = probs.argsort(descending=True)
            
            ranked_results = []
            for idx in sorted_indices:
                label = candidate_labels[idx.item()]
                score = probs[idx.item()].item()
                ranked_results.append((label, score))
                
            predictions.append(ranked_results)
            
        return predictions

    def evaluate_with_embeddings(self, query_embeddings: np.ndarray, query_metadata: pd.DataFrame,
                               database_embeddings: np.ndarray, database_metadata: pd.DataFrame,
                               device: str) -> Tuple[float, float]:
        """
        Evaluate using pre-computed embeddings with image-to-image similarity.
        
        Args:
            query_embeddings: (n_queries, embedding_dim) array
            query_metadata: DataFrame with metadata for query images
            database_embeddings: (n_database, embedding_dim) array  
            database_metadata: DataFrame with metadata for database images
            device: Device string (for compatibility)
            
        Returns:
            Tuple of (loss, average_precision)
        """
        precision_scores = {k: [] for k in self.k_values}
        
        # Limit number of queries if n_samples is specified
        n_queries = min(self.n_samples, len(query_embeddings)) if self.n_samples else len(query_embeddings)
        
        # Calculate similarities for all queries at once
        similarities = self.get_image_to_image_similarities(query_embeddings[:n_queries], database_embeddings)
        
        query_iterator = range(n_queries)
        if self.show_progress:
            query_iterator = tqdm(query_iterator, desc="Precision@K Evaluation (Image-to-Image)")
        
        for query_idx in query_iterator:
            query_species = query_metadata.iloc[query_idx]['species']
            query_similarities = similarities[query_idx]
            
            # Sort database by similarity descending
            sorted_indices = np.argsort(query_similarities)[::-1]
            ranked_species = [database_metadata.iloc[idx]['species'] for idx in sorted_indices]
            
            # Calculate precision@k for each k value
            for k in self.k_values:
                precision_k = self.precision_at_k(query_species, ranked_species, k)
                precision_scores[k].append(precision_k)
        
        # Calculate average precision scores
        avg_precision_scores = {}
        for k in self.k_values:
            if precision_scores[k]:
                avg_precision_scores[k] = sum(precision_scores[k]) / len(precision_scores[k])
            else:
                avg_precision_scores[k] = 0.0
                
        # Print individual precision scores
        print(f"\nEvaluated {n_queries} queries:")
        for k, score in avg_precision_scores.items():
            print(f"Precision@{k}: {score:.4f}")
            
        # Return average of all precision scores as the main metric
        overall_precision = sum(avg_precision_scores.values()) / len(avg_precision_scores) if avg_precision_scores else 0.0
        
        # No training loss for evaluation
        return 0.0, overall_precision
        
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """
        Evaluate model using Precision@K metrics.
        Returns (average_loss, average_precision) where average_precision is the mean of all k values.
        """
        all_labels = data_loader.get_labels()
        precision_scores = {k: [] for k in self.k_values}
        
        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="Precision@K Evaluation")
            
        total_queries = 0
        
        for query_images, true_labels, unique_ids in data_iter:
            if not query_images:
                continue
                
            # Get ranked predictions for each query image
            batch_predictions = self.get_ranked_predictions(model, query_images, all_labels)
            
            for i, (query_image, true_label) in enumerate(zip(query_images, true_labels)):
                ranked_results = batch_predictions[i]
                ranked_species = [label for label, _ in ranked_results]
                
                # Calculate precision@k for each k value
                for k in self.k_values:
                    precision_k = self.precision_at_k(true_label, ranked_species, k)
                    precision_scores[k].append(precision_k)
                
                total_queries += 1
        
        # Calculate average precision scores
        avg_precision_scores = {}
        for k in self.k_values:
            if precision_scores[k]:
                avg_precision_scores[k] = sum(precision_scores[k]) / len(precision_scores[k])
            else:
                avg_precision_scores[k] = 0.0
                
        # Print individual precision scores
        print(f"\nEvaluated {total_queries} queries:")
        for k, score in avg_precision_scores.items():
            print(f"Precision@{k}: {score:.4f}")
            
        # Return average of all precision scores as the main metric
        overall_precision = sum(avg_precision_scores.values()) / len(avg_precision_scores) if avg_precision_scores else 0.0
        
        # No training loss for evaluation
        return 0.0, overall_precision
        
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Override to include k_values information"""
        info = super().get_evaluator_info()
        info["k_values"] = self.k_values
        return info 