from tqdm import tqdm
from typing import List

class ZeroShotEvaluator:
    
    def __init__(self, n_samples: int = 200, metrics: List[str] = None):
        self.n_samples = n_samples
        self.metrics = metrics or ["accuracy"]
    
    def evaluate(self, model, data_loader, device: str):
        n_correct = 0
        
        for imgs, labels_targ in tqdm(data_loader.get_batch_data(self.n_samples), desc="Zero-Shot Eval"):
            labels_pred, _ = model.zero_shot_classify(imgs, data_loader.zs_labels)
            n_correct += sum(p == t for p, t in zip(labels_pred, labels_targ))
        
        accuracy = n_correct / self.n_samples
        print(f"Correct: {n_correct}/{self.n_samples}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return 0.0, accuracy 