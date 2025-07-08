from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .base import BaseEvaluator
from PIL import Image
import numpy as np
import pandas as pd


class mAPEvaluator(BaseEvaluator):
    """
    Evaluator that calculates mean Average Precision (mAP) using binary relevance (species-level).
    Only exact species matches are considered relevant.
    This calculates mAP (mean Average Precision) across multiple queries.
    """
    
    def __init__(self, n_samples: int = 200, show_progress: bool = True, **kwargs):
        metric_names = ["mAP"]
        super().__init__(n_samples=n_samples, metrics=metric_names, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress
        
    def average_precision(self, query_species: str, ranked_species: List[str], 
                         true_relevant_count: int = None) -> float:
        """
        Calculate Average Precision (AP) using binary relevance (species-level).
        
        Args:
            query_species: True species of the query image
            ranked_species: List of species in ranked order of similarity
            true_relevant_count: True relevant count if provided
        
        Returns:
            AP score (0.0 to 1.0)
        """
        if len(ranked_species) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        # Use provided true_relevant_count if available
        if true_relevant_count is not None:
            total_relevant = true_relevant_count
        else:
            total_relevant = sum(1 for species in ranked_species if species == query_species)
        
        if total_relevant == 0:
            return 0.0  # No relevant items
        
        for i, species in enumerate(ranked_species):
            if species == query_species:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        ap_score = precision_sum / total_relevant
        
        return ap_score
        
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
            Tuple of (loss, mAP_score)
        """
        ap_scores = []
        
        # Limit number of queries if n_samples is specified
        n_queries = min(self.n_samples, len(query_embeddings)) if self.n_samples else len(query_embeddings)
        
        # Get all species in database for counting relevant items
        database_species = database_metadata['species'].tolist()
        
        # Calculate similarities for all queries at once
        similarities = self.get_image_to_image_similarities(query_embeddings[:n_queries], database_embeddings)
        
        query_iterator = range(n_queries)
        if self.show_progress:
            query_iterator = tqdm(query_iterator, desc="mAP Evaluation (Image-to-Image)")
        
        for query_idx in query_iterator:
            query_species = query_metadata.iloc[query_idx]['species']
            query_similarities = similarities[query_idx]
            
            # Sort database by similarity descending
            sorted_indices = np.argsort(query_similarities)[::-1]
            ranked_species = [database_metadata.iloc[idx]['species'] for idx in sorted_indices]
            
            # Calculate the number of relevant items in the database for this species
            true_relevant_count = sum(1 for species in database_species if species == query_species)
            
            # Calculate Average Precision for this query
            ap_score = self.average_precision(query_species, ranked_species, true_relevant_count)
            ap_scores.append(ap_score)
        
        # Calculate mean Average Precision (mAP)
        if ap_scores:
            map_score = sum(ap_scores) / len(ap_scores)
        else:
            map_score = 0.0
                
        # Print result
        print(f"\nEvaluated {n_queries} queries:")
        print(f"mAP (mean Average Precision): {map_score:.4f}")
            
        # No training loss for evaluation
        return 0.0, map_score
        
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """
        Evaluate model using mean Average Precision (mAP) metric.
        Returns (average_loss, mAP) where mAP is the mean Average Precision across all queries.
        """
        all_labels = data_loader.get_labels()
        ap_scores = []
        
        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="mAP Evaluation")
            
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
                
                # Calculate Average Precision for this query
                ap_score = self.average_precision(true_label, ranked_species, true_relevant_count)
                ap_scores.append(ap_score)
                
                total_queries += 1
        
        # Calculate mean Average Precision (mAP)
        if ap_scores:
            map_score = sum(ap_scores) / len(ap_scores)
        else:
            map_score = 0.0
                
        # Print result
        print(f"\nEvaluated {total_queries} queries:")
        print(f"mAP (mean Average Precision): {map_score:.4f}")
            
        # No training loss for evaluation
        return 0.0, map_score
        
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Override to include mAP information"""
        info = super().get_evaluator_info()
        info["metric_type"] = "mAP (mean Average Precision)"
        return info 