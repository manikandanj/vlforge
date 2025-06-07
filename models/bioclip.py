import torch
from typing import List
from PIL import Image
import open_clip
from .base import BaseVisionModel


class BioCLIPModel(BaseVisionModel):
    
    def __init__(self, device: str = "cuda", model_name: str = "hf-hub:imageomics/bioclip", **kwargs):
        self.model_name_openclip = model_name
        super().__init__(device=device, model_name=model_name, **kwargs)
    
    def _setup_model(self):
        self.model, _, self.preprocessor = open_clip.create_model_and_transforms(self.model_name_openclip)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name_openclip)
    
    def _encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        imgs_pp = torch.stack([self.preprocessor(img) for img in images]).to(self.device)
        return self.model.encode_image(imgs_pp)
    
    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        text_tokens = self.tokenizer(texts).to(self.device)
        return self.model.encode_text(text_tokens)
    
    @property
    def model_name(self) -> str:
        return f"BioCLIP_{self.model_name_openclip.split('/')[-1]}" 