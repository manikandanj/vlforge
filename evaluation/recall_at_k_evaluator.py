from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .base import BaseEvaluator
from PIL import Image
import numpy as np
import pandas as pd


class RecallAtKEvaluator(BaseEvaluator):
    """
    Evaluator that calculates Recall@K using binary relevance (species-level).
    Only exact species matches are considered relevant.
    """
    
    def __init__(self, n_samples: int = 200, k_values: List[int] = None, 
                 show_progress: bool = True, **kwargs):
        self.k_values = k_values or [1, 5, 10, 15]
        metric_names = [f"recall@{k}" for k in self.k_values]
        super().__init__(n_samples=n_samples, metrics=metric_names, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress
        
    def recall_at_k(self, query_species: str, ranked_species: List[str], 
                   all_species_in_db: List[str], k: int, true_relevant_count: int = None) -> float:
        """
        Calculate Recall@k using binary relevance (species-level).
        
        Args:
            query_species: True species of the query image
            ranked_species: List of species in ranked order of similarity
            all_species_in_db: All species present in the database
            k: Number of top results to consider
            true_relevant_count: True relevant count if provided
        
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if k <= 0 or len(ranked_species) == 0:
            return 0.0
        
        # Use provided true_relevant_count if available
        if true_relevant_count is not None:
            total_relevant = true_relevant_count
        else:
            total_relevant = sum(1 for species in all_species_in_db if species == query_species)
        
        if total_relevant == 0:
            return 0.0  # No relevant items in database
        
        top_k_species = ranked_species[:k]
        relevant_retrieved = sum(1 for species in top_k_species if species == query_species)
        recall = relevant_retrieved / total_relevant
        
        return recall
        
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
            Tuple of (loss, average_recall)
        """
        recall_scores = {k: [] for k in self.k_values}
        
        # Limit number of queries if n_samples is specified
        n_queries = min(self.n_samples, len(query_embeddings)) if self.n_samples else len(query_embeddings)
        
        # Get all species in database for counting relevant items
        database_species = database_metadata['species'].tolist()
        
        # Calculate similarities for all queries at once
        similarities = self.get_image_to_image_similarities(query_embeddings[:n_queries], database_embeddings)
        
        query_iterator = range(n_queries)
        if self.show_progress:
            query_iterator = tqdm(query_iterator, desc="Recall@K Evaluation (Image-to-Image)")
        
        for query_idx in query_iterator:
            query_species = query_metadata.iloc[query_idx]['species']
            query_similarities = similarities[query_idx]
            
            # Sort database by similarity descending
            sorted_indices = np.argsort(query_similarities)[::-1]
            ranked_species = [database_metadata.iloc[idx]['species'] for idx in sorted_indices]
            
            # Calculate the number of relevant items in the database for this species
            true_relevant_count = sum(1 for species in database_species if species == query_species)
            
            # Calculate recall@k for each k value
            for k in self.k_values:
                recall_k = self.recall_at_k(query_species, ranked_species, database_species, k, true_relevant_count)
                recall_scores[k].append(recall_k)
        
        # Calculate average recall scores
        avg_recall_scores = {}
        for k in self.k_values:
            if recall_scores[k]:
                avg_recall_scores[k] = sum(recall_scores[k]) / len(recall_scores[k])
            else:
                avg_recall_scores[k] = 0.0
                
        # Print individual recall scores
        print(f"\nEvaluated {n_queries} queries:")
        for k, score in avg_recall_scores.items():
            print(f"Recall@{k}: {score:.4f}")
            
        # Return average of all recall scores as the main metric
        overall_recall = sum(avg_recall_scores.values()) / len(avg_recall_scores) if avg_recall_scores else 0.0
        
        # No training loss for evaluation
        return 0.0, overall_recall
        
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """
        Evaluate model using Recall@K metrics.
        Returns (average_loss, average_recall) where average_recall is the mean of all k values.
        """
        all_labels = data_loader.get_labels()
        recall_scores = {k: [] for k in self.k_values}
        
        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="Recall@K Evaluation")
            
        total_queries = 0
        
        for query_images, true_labels, unique_ids in data_iter:
            if not query_images:
                continue
                
            # Get ranked predictions for each query image
            batch_predictions = self.get_ranked_predictions(model, query_images, all_labels)
            
            for i, (query_image, true_label) in enumerate(zip(query_images, true_labels)):
                ranked_results = batch_predictions[i]
                ranked_species = [label for label, _ in ranked_results]
                
                # Calculate the number of relevant items in the database for this species
                true_relevant_count = sum(1 for species in all_labels if species == true_label)
                
                # Calculate recall@k for each k value
                for k in self.k_values:
                    recall_k = self.recall_at_k(true_label, ranked_species, all_labels, k, true_relevant_count)
                    recall_scores[k].append(recall_k)
                
                total_queries += 1
        
        # Calculate average recall scores
        avg_recall_scores = {}
        for k in self.k_values:
            if recall_scores[k]:
                avg_recall_scores[k] = sum(recall_scores[k]) / len(recall_scores[k])
            else:
                avg_recall_scores[k] = 0.0
                
        # Print individual recall scores
        print(f"\nEvaluated {total_queries} queries:")
        for k, score in avg_recall_scores.items():
            print(f"Recall@{k}: {score:.4f}")
            
        # Return average of all recall scores as the main metric
        overall_recall = sum(avg_recall_scores.values()) / len(avg_recall_scores) if avg_recall_scores else 0.0
        
        # No training loss for evaluation
        return 0.0, overall_recall
        
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Override to include k_values information"""
        info = super().get_evaluator_info()
        info["k_values"] = self.k_values
        return info 