import math
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from .base import BaseEvaluator
from PIL import Image


class NDCGEvaluator(BaseEvaluator):
    """
    Evaluator that calculates nDCG (Normalized Discounted Cumulative Gain) 
    using taxonomic hierarchy-based relevance scoring.
    """
    
    def __init__(self, n_samples: int = 200, metrics: List[str] = None, 
                 k_values: List[int] = None, show_progress: bool = True, **kwargs):
        self.k_values = k_values or [5, 10, 50]
        metric_names = [f"ndcg@{k}" for k in self.k_values]
        super().__init__(n_samples=n_samples, metrics=metrics or metric_names, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress
        self.frequency_maps = None
        self.species_to_taxonomy = {}  # Cache for species -> (genus, subfamily) mapping
        
    def _build_species_taxonomy_map(self, data_loader):
        """
        Build a mapping from species name to taxonomic information using the data loader's metadata.
        """
        self.species_to_taxonomy = {}
        
        # Access the metadata DataFrame from the butterfly data loader
        if hasattr(data_loader, 'df_meta'):
            df_meta = data_loader.df_meta
            for _, row in df_meta.iterrows():
                species = row['species']
                genus = row['genus'] if 'genus' in row else ""
                subfamily = row['subfamily'] if 'subfamily' in row else ""
                
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
        
    def calculate_dcg(self, relevance_scores: List[int], k: int) -> float:
        """Calculate Discounted Cumulative Gain for top-k results"""
        dcg = 0.0
        for i in range(min(k, len(relevance_scores))):
            if i == 0:
                dcg += relevance_scores[i]
            else:
                dcg += relevance_scores[i] / math.log2(i + 1)
        return dcg
        
    def calculate_idcg(self, relevance_scores: List[int], k: int) -> float:
        """Calculate Ideal DCG by sorting relevance scores in descending order"""
        sorted_scores = sorted(relevance_scores, reverse=True)
        return self.calculate_dcg(sorted_scores, k)
        
    def calculate_ndcg(self, relevance_scores: List[int], k: int) -> float:
        """Calculate Normalized DCG"""
        dcg = self.calculate_dcg(relevance_scores, k)
        idcg = self.calculate_idcg(relevance_scores, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
        
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
        
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        """
        Evaluate model using nDCG metrics.
        Returns (average_loss, average_ndcg) where average_ndcg is the mean of all k values.
        """
        # Get frequency maps from data loader for taxonomic information
        self.frequency_maps = data_loader.get_frequency_maps()
        
        # Build species to taxonomy mapping from metadata
        self._build_species_taxonomy_map(data_loader)
        
        all_labels = data_loader.get_labels()
        ndcg_scores = {k: [] for k in self.k_values}
        
        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="nDCG Evaluation")
            
        total_queries = 0
        
        for query_images, true_labels in data_iter:
            if not query_images:
                continue
                
            # Get ranked predictions for each query image
            batch_predictions = self.get_ranked_predictions(model, query_images, all_labels)
            
            for i, (query_image, true_label) in enumerate(zip(query_images, true_labels)):
                ranked_results = batch_predictions[i]
                
                # Calculate relevance scores for the ranked results
                relevance_scores = []
                for predicted_label, _ in ranked_results:
                    relevance_score = self.get_relevance_score(true_label, predicted_label)
                    relevance_scores.append(relevance_score)
                
                # Calculate nDCG for each k value
                for k in self.k_values:
                    ndcg_k = self.calculate_ndcg(relevance_scores, k)
                    ndcg_scores[k].append(ndcg_k)
                
                total_queries += 1
        
        # Calculate average nDCG scores
        avg_ndcg_scores = {}
        for k in self.k_values:
            if ndcg_scores[k]:
                avg_ndcg_scores[k] = sum(ndcg_scores[k]) / len(ndcg_scores[k])
            else:
                avg_ndcg_scores[k] = 0.0
                
        # Print individual nDCG scores
        print(f"\nEvaluated {total_queries} queries:")
        for k, score in avg_ndcg_scores.items():
            print(f"nDCG@{k}: {score:.4f}")
            
        # Return average of all nDCG scores as the main metric
        overall_ndcg = sum(avg_ndcg_scores.values()) / len(avg_ndcg_scores) if avg_ndcg_scores else 0.0
        
        # No training loss for evaluation
        return 0.0, overall_ndcg
        
    def get_evaluator_info(self) -> Dict[str, Any]:
        """Override to include k_values information"""
        info = super().get_evaluator_info()
        info["k_values"] = self.k_values
        return info 