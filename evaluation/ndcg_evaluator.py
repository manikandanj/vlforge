from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from .base import BaseEvaluator
from PIL import Image
import numpy as np
import pandas as pd


class NDCGEvaluator(BaseEvaluator):
    """
    Evaluator that calculates nDCG (Normalized Discounted Cumulative Gain) 
    using hierarchical taxonomic relevance scoring with exponential gain.
    
    Relevance levels:
    - Species match: 3 (Highly Relevant)
    - Genus match: 2 (Relevant)
    - Subfamily match: 1 (Partially Relevant)
    - No match: 0 (Irrelevant)
    
    Uses exponential gain: (2^relevance - 1) / log2(i + 2)
    Calculates mean nDCG across all queries and reports both individual k-values and overall mean.
    """
    
    def __init__(self, n_samples: int = 200, k_values: List[int] = None, 
                 primary_k: int = None, show_progress: bool = True, **kwargs):
        self.k_values = k_values or [1, 5, 15]
        self.primary_k = primary_k or max(self.k_values)  # Use highest k as primary by default
        metric_names = [f"ndcg@{k}" for k in self.k_values]
        super().__init__(n_samples=n_samples, metrics=metric_names, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress
        self.species_to_taxonomy = {}  # Cache for species -> (genus, subfamily) mapping
        
    def _build_species_taxonomy_map(self, data_loader):
        """
        Build a mapping from species name to taxonomic information using the data loader's metadata.
        """
        self.species_to_taxonomy = {}
        
        # Get all unique labels (species) from the data loader
        all_species = data_loader.get_labels()
        
        # For butterfly dataset, we need to get some sample unique_ids to build the taxonomy map
        # Sample a small batch to get representative metadata
        sample_metadata = []
        for images, labels, unique_ids in data_loader.get_batch_data(n_samples=100):
            metadata_batch = data_loader.get_metadata_for_ids(unique_ids)
            sample_metadata.extend(metadata_batch)
            break  # Just need one batch to build the map
        
        # Build species to taxonomy mapping from the metadata
        for metadata in sample_metadata:
            species = metadata.get('species', '')
            if species:
                genus = metadata.get('genus', '')
                subfamily = metadata.get('subfamily', '')
                
                # Store the taxonomic information for this species
                self.species_to_taxonomy[species] = {
                    'genus': genus,
                    'subfamily': subfamily
                }
        
    def get_relevance_score(self, query_species: str, result_species: str) -> int:
        """
        Calculate relevance score based on taxonomic hierarchy:
        - Species match: 3 (Highly Relevant)
        - Genus match: 2 (Relevant) 
        - Subfamily match: 1 (Partially Relevant)
        - No match: 0 (Irrelevant)
        """
        if query_species == result_species:
            return 3
            
        # Get taxonomic information from cached mapping
        query_taxonomy = self.species_to_taxonomy.get(query_species, {})
        result_taxonomy = self.species_to_taxonomy.get(result_species, {})
        
        query_genus = query_taxonomy.get('genus', '')
        result_genus = result_taxonomy.get('genus', '')
        query_subfamily = query_taxonomy.get('subfamily', '')
        result_subfamily = result_taxonomy.get('subfamily', '')
        
        # Check genus match
        if query_genus and result_genus and query_genus == result_genus:
            return 2
            
        # Check subfamily match (using as family-level classification)
        if query_subfamily and result_subfamily and query_subfamily == result_subfamily:
            return 1
            
        return 0
        
    def ndcg_at_k(self, query_species: str, ranked_results: List[Tuple[str, float]], k: int) -> float:
        """
        Calculate nDCG@k using hierarchical relevance scoring (multi-level taxonomy).
        Uses exponential gain: (2^relevance - 1) / log2(i + 2)
        
        Args:
            query_species: True species of the query image
            ranked_results: List of (species, similarity_score) tuples in ranked order
            k: Number of top results to consider
        
        Returns:
            nDCG@k score (0.0 to 1.0)
        """
        if k <= 0 or len(ranked_results) == 0:
            return 0.0
        
        # Get relevance scores for top-k results
        relevance_scores = []
        prediction_scores = []
        
        for i, (species, similarity_score) in enumerate(ranked_results[:k]):
            relevance = self.get_relevance_score(query_species, species)
            relevance_scores.append(relevance)
            prediction_scores.append(similarity_score)
        
        # Calculate DCG using exponential gain
        dcg = 0.0
        for i, relevance in enumerate(relevance_scores):
            # Using exponential gain instead of linear gain
            dcg += (2**relevance - 1) / np.log2(i + 2)
            
        # Calculate ideal DCG (best possible ordering)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0.0
        for i, relevance in enumerate(ideal_relevance):
            # Using exponential gain instead of linear gain
            idcg += (2**relevance - 1) / np.log2(i + 2)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        
        return ndcg
        
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
        Optimized for large datasets.
        
        Args:
            query_embeddings: (n_queries, embedding_dim) array
            query_metadata: DataFrame with metadata for query images
            database_embeddings: (n_database, embedding_dim) array  
            database_metadata: DataFrame with metadata for database images
            device: Device string (for compatibility)
            
        Returns:
            Tuple of (loss, primary_ndcg)
        """
        # Build species to taxonomy mapping from metadata
        self._build_species_taxonomy_map_from_metadata(database_metadata)
        
        ndcg_scores = {k: [] for k in self.k_values}
        
        # Determine number of queries to evaluate (use all from split or limit with n_samples)
        n_queries = self._get_query_count(len(query_embeddings))
        
        # Pre-convert to numpy arrays for faster access
        query_species = query_metadata['species'].values[:n_queries]
        database_species = database_metadata['species'].values
        
        # Calculate similarities for all queries at once
        similarities = self.get_image_to_image_similarities(query_embeddings[:n_queries], database_embeddings)
        
        # Get max k for efficient top-k computation
        max_k = max(self.k_values) if self.k_values else 50
        top_k_indices = self.get_top_k_results_fast(similarities, database_species, k=max_k)
        
        query_iterator = range(n_queries)
        if self.show_progress:
            query_iterator = tqdm(query_iterator, desc="nDCG Evaluation (Image-to-Image, Hierarchical)")
        
        for query_idx in query_iterator:
            query_species_name = query_species[query_idx]
            top_k_db_indices = top_k_indices[query_idx]
            
            # Get species and similarities for top-k results using fast numpy indexing
            ranked_species = database_species[top_k_db_indices]
            ranked_similarities = similarities[query_idx, top_k_db_indices]
            
            # Create ranked results for nDCG computation
            ranked_results = [(species, sim_score) for species, sim_score in zip(ranked_species, ranked_similarities)]
            
            # Calculate nDCG for each k value using hierarchical implementation
            for k in self.k_values:
                k_limited = min(k, len(ranked_results))
                ndcg_k = self.ndcg_at_k(query_species_name, ranked_results[:k_limited], k_limited)
                ndcg_scores[k].append(ndcg_k)
        
        # Calculate mean nDCG scores across all queries
        avg_ndcg_scores = {}
        for k in self.k_values:
            if ndcg_scores[k]:
                avg_ndcg_scores[k] = sum(ndcg_scores[k]) / len(ndcg_scores[k])
            else:
                avg_ndcg_scores[k] = 0.0
        
        # Print individual nDCG scores following standard IR practices
        print(f"\nEvaluated {n_queries} queries (Hierarchical Taxonomic Relevance):")
        for k, score in avg_ndcg_scores.items():
            print(f"nDCG@{k}: {score:.4f}")
        
        # Return primary k value as main metric
        primary_ndcg = avg_ndcg_scores.get(self.primary_k, 0.0)
        
        # No training loss for evaluation
        return 0.0, primary_ndcg

    def _build_species_taxonomy_map_from_metadata(self, metadata_df: pd.DataFrame):
        """
        Build a mapping from species name to taxonomic information using the metadata DataFrame.
        """
        self.species_to_taxonomy = {}
        
        # Build species to taxonomy mapping from the metadata
        for _, row in metadata_df.iterrows():
            species = row.get('species', '')
            if species:
                genus = row.get('genus', '')
                subfamily = row.get('subfamily', '')
                
                # Store the taxonomic information for this species
                self.species_to_taxonomy[species] = {
                    'genus': genus,
                    'subfamily': subfamily
                }
        
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """
        Evaluate model using hierarchical nDCG metrics with exponential gain.
        Returns (average_loss, primary_ndcg) where primary_ndcg is the mean nDCG@primary_k across all queries.
        Individual k-values are reported separately following standard IR practices.
        """
        # Build species to taxonomy mapping from metadata
        self._build_species_taxonomy_map(data_loader)
        
        all_labels = data_loader.get_labels()
        ndcg_scores = {k: [] for k in self.k_values}
        
        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="nDCG Evaluation (Hierarchical)")
            
        total_queries = 0
        
        for query_images, true_labels, unique_ids in data_iter:
            if not query_images:
                continue
                
            # Get ranked predictions for each query image
            batch_predictions = self.get_ranked_predictions(model, query_images, all_labels)
            
            for i, (query_image, true_label) in enumerate(zip(query_images, true_labels)):
                ranked_results = batch_predictions[i]
                
                # Calculate nDCG for each k value using hierarchical implementation
                for k in self.k_values:
                    ndcg_k = self.ndcg_at_k(true_label, ranked_results, k)
                    ndcg_scores[k].append(ndcg_k)
                
                total_queries += 1
        
        # Calculate mean nDCG scores across all queries
        avg_ndcg_scores = {}
        for k in self.k_values:
            if ndcg_scores[k]:
                avg_ndcg_scores[k] = sum(ndcg_scores[k]) / len(ndcg_scores[k])
            else:
                avg_ndcg_scores[k] = 0.0
                
        # Print individual nDCG scores (mean across queries for each k)
        print(f"\nEvaluated {total_queries} queries:")
        print("Hierarchical nDCG (exponential gain):")
        for k, score in avg_ndcg_scores.items():
            print(f"Mean nDCG@{k}: {score:.4f}")
        
        # Return primary k-value as main metric (standard practice)
        primary_ndcg = avg_ndcg_scores.get(self.primary_k, 0.0)
        print(f"Primary metric (nDCG@{self.primary_k}): {primary_ndcg:.4f}")
            
        # No training loss for evaluation
        return 0.0, primary_ndcg
        
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Override to include k_values and implementation details"""
        info = super().get_evaluator_info()
        info["k_values"] = self.k_values
        info["primary_k"] = self.primary_k
        info["gain_type"] = "exponential"
        info["relevance_levels"] = "Species=3, Genus=2, Subfamily=1, None=0"
        info["mean_calculation"] = "across queries for each k-value separately"
        return info 