from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseEvaluator(ABC):
    
    def __init__(self, n_samples: int = 200, metrics: List[str] = None, **kwargs):
        self.n_samples = n_samples
        self.metrics = metrics or ["accuracy"]
        self.config = kwargs
        self.evaluator_name = self.__class__.__name__
        
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
    
    
