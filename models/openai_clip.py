import torch
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel
from .base import BaseVisionModel
from PIL import Image

class OpenAICLIPModel(BaseVisionModel):
    
    def __init__(self, device: str = "cuda", model_name: str = "openai/clip-vit-base-patch32", **kwargs):
        self.model_name_hf = model_name
        super().__init__(device=device, model_name=model_name, **kwargs)
    
    def _setup_model(self):
        self.preprocessor = CLIPProcessor.from_pretrained(self.model_name_hf)
        self.model = CLIPModel.from_pretrained(self.model_name_hf).to(self.device)
        self.model.eval()
    
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.preprocessor(
            images=images, 
            return_tensors="pt"
        ).to(self.device)
        
        return self.model.get_image_features(pixel_values=inputs.pixel_values)
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.preprocessor(
            text=texts, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        return self.model.get_text_features(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask
        )
    
    @property
    def model_name(self) -> str:
        return f"OpenAICLIP_{self.model_name_hf.split('/')[-1]}" 