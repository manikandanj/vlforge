from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BaseEvaluator(ABC):
    
    def __init__(self, n_samples: int = 200, metrics: List[str] = None, 
                 data_split_filter: Optional[Dict[str, Any]] = None, **kwargs):
        self.n_samples = n_samples
        self.metrics = metrics or ["accuracy"]
        self.config = kwargs
        self.evaluator_name = self.__class__.__name__
        
        # Foundation for flexible data splits (e.g., species with <10 instances)
        # This will be extended in future iterations
        self.data_split_filter = data_split_filter or {}
        
    def _apply_data_split_filter(self, images, labels, unique_ids, metadata=None):
        """
        Apply data split filter to limit evaluation to specific subsets.
        This is a foundation for future implementation of split-based evaluation.
        
        Args:
            images: List of images
            labels: List of labels  
            unique_ids: List of unique IDs
            metadata: Optional metadata for filtering
            
        Returns:
            Filtered (images, labels, unique_ids) tuple
        """
        # Currently just returns all data, but provides foundation for:
        # - species_min_count: only evaluate species with >= N instances
        # - species_max_count: only evaluate species with <= N instances  
        # - genus_filter: only evaluate specific genera
        # - subfamily_filter: only evaluate specific subfamilies
        # - custom_filter_fn: apply custom filtering function
        
        if not self.data_split_filter:
            return images, labels, unique_ids
            
        # TODO: Implement filtering logic when needed
        # For now, return all data
        return images, labels, unique_ids
    
    def get_image_to_image_similarities(self, query_embeddings: np.ndarray, 
                                      database_embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and database embeddings.
        
        Args:
            query_embeddings: (n_queries, embedding_dim) array
            database_embeddings: (n_database, embedding_dim) array
            
        Returns:
            (n_queries, n_database) similarity matrix
        """
        return cosine_similarity(query_embeddings, database_embeddings)
    
    def get_ranked_results_from_similarities(self, similarities: np.ndarray, 
                                           database_metadata: pd.DataFrame) -> List[List[Tuple[str, float]]]:
        """
        Convert similarity matrix to ranked results.
        
        Args:
            similarities: (n_queries, n_database) similarity matrix
            database_metadata: DataFrame with metadata for database images
            
        Returns:
            List of lists: each inner list contains (species, similarity_score) tuples 
            sorted by similarity for one query image.
        """
        ranked_results = []
        
        for query_idx in range(similarities.shape[0]):
            query_similarities = similarities[query_idx]
            
            # Sort by similarity descending
            sorted_indices = np.argsort(query_similarities)[::-1]
            
            query_ranked_results = []
            for db_idx in sorted_indices:
                species = database_metadata.iloc[db_idx]['species']
                similarity_score = query_similarities[db_idx]
                query_ranked_results.append((species, similarity_score))
                
            ranked_results.append(query_ranked_results)
            
        return ranked_results
        
    @abstractmethod
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """Original evaluation method using model and data_loader (for backward compatibility)"""
        pass
    
    def evaluate_with_embeddings(self, query_embeddings: np.ndarray, query_metadata: pd.DataFrame,
                               database_embeddings: np.ndarray, database_metadata: pd.DataFrame,
                               device: str) -> Tuple[float, float]:
        """
        New evaluation method using pre-computed embeddings.
        Default implementation uses image-to-image similarity.
        Subclasses should override this method.
        
        Args:
            query_embeddings: (n_queries, embedding_dim) array
            query_metadata: DataFrame with metadata for query images
            database_embeddings: (n_database, embedding_dim) array  
            database_metadata: DataFrame with metadata for database images
            device: Device string (for compatibility)
            
        Returns:
            Tuple of (loss, metric_value)
        """
        # Default implementation - subclasses should override
        print("Warning: Using default embedding evaluation - subclasses should override this method")
        
        # Calculate similarities
        similarities = self.get_image_to_image_similarities(query_embeddings, database_embeddings)
        
        # Get ranked results
        ranked_results = self.get_ranked_results_from_similarities(similarities, database_metadata)
        
        return 0.0, 0.0  # Placeholder return
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "config": self.config
        }
    
    
