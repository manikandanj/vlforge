from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class BaseTrainer(ABC):
    """Just added a blank class. To be implemented"""
    
    def __init__(self, epochs: int = 10, **kwargs):
        self.epochs = epochs
        self.config = kwargs
        
    @abstractmethod
    def train(self):
        pass