import torch
from typing import List
from PIL import Image
import open_clip
from .base import BaseVisionModel
from utils.logging_config import get_logger


class FineTunedBioCLIPModel(BaseVisionModel):
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", 
                 base_model_name: str = "hf-hub:imageomics/bioclip", **kwargs):
        """
        Initialize a fine-tuned BioCLIP model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the .pt file containing fine-tuned weights
            device: Device to load the model on
            base_model_name: Base BioCLIP model name to use as architecture
            **kwargs: Additional arguments passed to parent class
        """
        self.checkpoint_path = checkpoint_path
        self.base_model_name = base_model_name
        super().__init__(device=device, model_name=f"FineTuned_{base_model_name}", **kwargs)
    
    def _setup_model(self):
        """Setup the model by loading base architecture and fine-tuned weights."""
        # Load the base model architecture
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(self.base_model_name)
        
        # Load the fine-tuned weights
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
            
            # Load the weights
            self.model.load_state_dict(state_dict, strict=False)
            logger = get_logger()
            logger.info(f"Successfully loaded fine-tuned weights from {self.checkpoint_path}")
            
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Could not load fine-tuned weights from {self.checkpoint_path}: {e}")
            logger.warning("Using base BioCLIP model instead")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup tokenizer
        self.tokenizer = open_clip.get_tokenizer(self.base_model_name)
    
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode a list of images into embeddings."""
        imgs_pp = torch.stack([self.preprocessor(img) for img in images]).to(self.device)
        return self.model.encode_image(imgs_pp)
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts into embeddings."""
        text_tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(text_tokens)
    
    @property
    def model_name(self) -> str:
        """Return a descriptive name for this model."""
        return f"FineTunedBioCLIP_{self.base_model_name.split('/')[-1]}"
    
    def get_checkpoint_info(self) -> dict:
        """Get information about the loaded checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            info = {
                "checkpoint_path": self.checkpoint_path,
                "base_model": self.base_model_name,
                "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "state_dict_only"
            }
            
            # Try to get additional info if available
            if isinstance(checkpoint, dict):
                if 'epoch' in checkpoint:
                    info['epoch'] = checkpoint['epoch']
                if 'best_score' in checkpoint:
                    info['best_score'] = checkpoint['best_score']
                if 'optimizer' in checkpoint:
                    info['has_optimizer_state'] = True
                    
            return info
        except Exception as e:
            return {"error": f"Could not load checkpoint info: {e}"} 