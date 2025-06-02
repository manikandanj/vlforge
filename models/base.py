from abc import ABC, abstractmethod
from typing import List, Tuple
import torch

class BaseCLIPModel(ABC):
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def zero_shot_classify(self, imgs: List, zs_labels: List[str]) -> Tuple[List[str], List[float]]:
        pass 