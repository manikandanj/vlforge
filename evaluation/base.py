import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_config import get_logger


class BaseEvaluator(ABC):
    
    def __init__(self, n_samples: Optional[int] = 200, metrics: List[str] = None, 
                 data_split_filter: Optional[Dict[str, Any]] = None, use_gpu: bool = True, **kwargs):
        self.n_samples = n_samples  # If None/null, use all available queries
        self.metrics = metrics or ["accuracy"]
        self.config = kwargs
        self.evaluator_name = self.__class__.__name__
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
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
        Optimized for large datasets using chunked processing.
        Uses GPU acceleration if available.
        
        Args:
            query_embeddings: (n_queries, embedding_dim) array
            database_embeddings: (n_database, embedding_dim) array
            
        Returns:
            (n_queries, n_database) similarity matrix
        """
        if self.use_gpu:
            return self._get_similarities_gpu(query_embeddings, database_embeddings)
        else:
            return self._get_similarities_cpu(query_embeddings, database_embeddings)
    
    def _get_similarities_gpu(self, query_embeddings: np.ndarray, 
                             database_embeddings: np.ndarray) -> np.ndarray:
        """GPU-accelerated similarity computation using PyTorch."""
        logger = get_logger()
        n_queries, n_database = query_embeddings.shape[0], database_embeddings.shape[0]
        
        logger.info(f"Computing similarities on GPU ({n_queries}×{n_database}={n_queries*n_database:,})")
        
        # Convert to PyTorch tensors and move to GPU
        query_tensor = torch.tensor(query_embeddings, dtype=torch.float32, device=self.device)
        db_tensor = torch.tensor(database_embeddings, dtype=torch.float32, device=self.device)
        
        # Normalize embeddings for cosine similarity
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
        db_tensor = torch.nn.functional.normalize(db_tensor, p=2, dim=1)
        
        # Compute cosine similarity: query @ database.T
        similarities = torch.mm(query_tensor, db_tensor.t())
        
        # Convert back to numpy
        return similarities.cpu().numpy()
    
    def _get_similarities_cpu(self, query_embeddings: np.ndarray, 
                             database_embeddings: np.ndarray) -> np.ndarray:
        """CPU-based similarity computation with chunking for large datasets."""
        n_queries = query_embeddings.shape[0]
        n_database = database_embeddings.shape[0]
        

    
    def get_top_k_results_fast(self, similarities: np.ndarray, database_species: np.ndarray, 
                              k: int = 50) -> np.ndarray:
        """
        Fast method to get top-K results without full sorting.
        Uses GPU acceleration if available.
        
        Args:
            similarities: (n_queries, n_database) similarity matrix
            database_species: (n_database,) array of species names
            k: Number of top results to return
            
        Returns:
            (n_queries, k) array of top-k species indices
        """
        if self.use_gpu:
            return self._get_top_k_gpu(similarities, k)
        else:
            return self._get_top_k_cpu(similarities, k)
    
    def _get_top_k_gpu(self, similarities: np.ndarray, k: int) -> np.ndarray:
        """GPU-accelerated top-k selection using PyTorch."""
        n_queries, n_database = similarities.shape
        k = min(k, n_database)
        
        # Convert to PyTorch tensor and move to GPU
        sim_tensor = torch.tensor(similarities, dtype=torch.float32, device=self.device)
        
        # Get top-k indices using PyTorch (much faster on GPU)
        _, top_k_indices = torch.topk(sim_tensor, k, dim=1)
        
        # Convert back to numpy
        return top_k_indices.cpu().numpy()
    
    def _get_top_k_cpu(self, similarities: np.ndarray, k: int) -> np.ndarray:
        """CPU-based top-k selection using numpy."""
        # Use argpartition for faster top-k (O(n) vs O(n log n) for full sort)
        # Note: argpartition doesn't fully sort, just partitions around k-th element
        n_queries, n_database = similarities.shape
        k = min(k, n_database)  # Don't request more than available
        
        # Get top-k indices for all queries at once
        top_k_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
        
        # For each query, sort only the top-k results
        for i in range(n_queries):
            top_k_similarities = similarities[i, top_k_indices[i]]
            sorted_order = np.argsort(top_k_similarities)[::-1]  # Sort in descending order
            top_k_indices[i] = top_k_indices[i][sorted_order]
            
        return top_k_indices
    
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
    
    def _get_query_count(self, total_queries: int) -> int:
        """
        Determine how many queries to evaluate based on n_samples.
        
        Args:
            total_queries: Total number of available queries
            
        Returns:
            Number of queries to actually evaluate
        """
        if self.n_samples is None:
            return total_queries  # Use all available queries
        else:
            return min(self.n_samples, total_queries)
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "config": self.config
        }
    
    
