import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image


class BaseVisionModel(ABC):
   
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocessor = None
        self.model_config = kwargs
        self._setup_model()
        
    @abstractmethod
    def _setup_model(self):
        pass
    
    @abstractmethod
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        pass
    
    def zero_shot_classify(self, images, labels):
        return self._compute_predictions(images, labels)

    def _compute_predictions(self, images, labels):
        with torch.no_grad():
            similarities = self.compute_similarity(images, labels) 
            
        probs = similarities.softmax(dim=-1)
        idxs_pred = probs.argmax(dim=-1)
        scores = probs[torch.arange(len(similarities)), idxs_pred].tolist()
        predicted_labels = [labels[i] for i in idxs_pred.tolist()]
        
        return predicted_labels, scores
    
    def get_image_embeddings(self, images: List[Image.Image]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self._encode_images(images)
            return F.normalize(embeddings, dim=-1)
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self._encode_text(texts)
            return F.normalize(embeddings, dim=-1)
    
    """
    Compute the similarity between a list of images and a list of texts
    """
    def compute_similarity(self, images: List[Image.Image], texts: List[str]) -> torch.Tensor:
        img_embeddings = self.get_image_embeddings(images)
        txt_embeddings = self.get_text_embeddings(texts)
        return img_embeddings @ txt_embeddings.T
    
    @property
    def model_name(self) -> str:
        return self.__class__.__name__
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "config": self.model_config
        } 