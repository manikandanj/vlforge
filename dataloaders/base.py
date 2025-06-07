from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Optional, Dict, Any
from PIL import Image


class BaseDataLoader(ABC):
    """Abstract base class for data loaders providing a unified interface"""
    
    def __init__(self, batch_size: int = 32, **kwargs):
        self.batch_size = batch_size
        self.config = kwargs
        self._setup_data()
        
    @abstractmethod
    def _setup_data(self):
        pass
    
    @abstractmethod
    def load_image(self, image_identifier: str) -> Image.Image:
        pass
    
    @abstractmethod
    def get_labels(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_batch_data(self, n_samples: int) -> Generator[Tuple[List[Image.Image], List[str]], None, None]:
        pass
    
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "loader_name": self.__class__.__name__,
            "batch_size": self.batch_size,
            "num_labels": len(self.get_labels()),
            "config": self.config
        }
