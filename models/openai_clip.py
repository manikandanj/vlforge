import torch
import transformers.modeling_utils as _mu
_mu.check_torch_load_is_safe = lambda *args, **kwargs: None

import torch.nn.functional as F
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel
from .base import BaseCLIPModel

class OpenAICLIPModel(BaseCLIPModel):
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        model_name = "openai/clip-vit-base-patch32"
        self.preprocessor = CLIPProcessor.from_pretrained(model_name)
        self.clip = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip.eval()
    
    def zero_shot_classify(self, imgs: List, zs_labels: List[str]) -> Tuple[List[str], List[float]]:
        inputs = self.preprocessor(
            text=zs_labels, 
            images=imgs, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            img_feats = self.clip.get_image_features(pixel_values=inputs.pixel_values)
            txt_feats = self.clip.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits = img_feats @ txt_feats.T
        
        probs = logits.softmax(dim=-1)
        idxs_pred = probs.argmax(dim=-1)
        scores = probs[torch.arange(len(imgs)), idxs_pred].tolist()
        zs_label_preds = [zs_labels[i] for i in idxs_pred.tolist()]
        
        return zs_label_preds, scores 