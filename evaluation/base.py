from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Callable


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
        
    @abstractmethod
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        pass
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
            "config": self.config
        }
    
    
