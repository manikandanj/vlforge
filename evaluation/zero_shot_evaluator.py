from tqdm import tqdm
from typing import List, Tuple
from .base import BaseEvaluator
from PIL import Image


class ZeroShotEvaluator(BaseEvaluator):
    
    def __init__(self, n_samples: int = 200, metrics: List[str] = None, 
                 show_progress: bool = True, **kwargs):
        super().__init__(n_samples=n_samples, metrics=metrics, 
                        show_progress=show_progress, **kwargs)
        self.show_progress = show_progress

    def zero_shot_classify(self, model, images: List[Image.Image], labels: List[str]) -> Tuple[List[str], List[float]]:
        """
        Perform zero-shot classification: evaluate model predictions without training examples.
        """
        return model.compute_predictions(images, labels)
    
    
    def evaluate(self, model, data_loader, device: str) -> Tuple[float, float]:
        n_correct = 0
        n_total = 0

        data_iter = data_loader.get_batch_data(self.n_samples)
        if self.show_progress:
            data_iter = tqdm(data_iter, desc="Zero-Shot Eval")
        
        for images, target_labels, unique_ids in data_iter:
            if not images:
                continue
            ## data_loader.get_labels() returns all possible classes in the dataset whereas target_labels has the ground truth
            predicted_labels, scores = self.zero_shot_classify(model, images, data_loader.get_labels())
            
            for pred, target in zip(predicted_labels, target_labels):
                if pred == target:
                    n_correct += 1
                n_total += 1
        
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        
        # No training loss for zero-shot
        return 0.0, accuracy 