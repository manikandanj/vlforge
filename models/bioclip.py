import torch
import torch.nn.functional as F
from typing import List, Tuple
import open_clip
from .base import BaseCLIPModel

class BioCLIPModel(BaseCLIPModel):
    
    def __init__(self, device: str = "cuda"):
        super().__init__(device)
        self.clip, _, self.preprocessor = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
        self.clip.to(self.device)
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    
    def zero_shot_classify(self, imgs: List, zs_labels: List[str]) -> Tuple[List[str], List[float]]:
        imgs_pp = torch.stack([self.preprocessor(img) for img in imgs]).to(self.device)
        zs_label_tokens = self.tokenizer(zs_labels).to(self.device)
        
        with torch.no_grad():
            img_feats = self.clip.encode_image(imgs_pp)
            txt_feats = self.clip.encode_text(zs_label_tokens)
        
        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        logits = img_feats @ txt_feats.T
        
        probs = logits.softmax(dim=-1)
        idxs_pred = probs.argmax(dim=-1)
        scores = probs[torch.arange(len(imgs)), idxs_pred].tolist()
        zs_label_preds = [zs_labels[i] for i in idxs_pred.tolist()]
        
        return zs_label_preds, scores 