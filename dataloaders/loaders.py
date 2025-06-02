import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator
from .config import DatasetConfig

class BiologicalDataLoader:
    
    def __init__(self, base_dir: str, group: str, batch_size: int = 32, num_workers: int = 4):
        self.config = DatasetConfig(base_dir, group, batch_size, num_workers)
        self.df_meta = pd.read_csv(self.config.metadata_path)
        self.zs_labels = sorted(self.df_meta["scientificName"].dropna().unique())
        
        chunk_dirs = list(self.config.img_dir.glob("chunk_*"))
        if not chunk_dirs:
            raise ValueError(f"No chunk directories found in {self.config.img_dir}")
        print(f"Found {len(chunk_dirs)} chunk directories")
    
    def load_img(self, filename_img: str) -> Image.Image:
        for chunk_dir in self.config.img_dir.glob("chunk_*"):
            img_path = chunk_dir / filename_img
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
        raise FileNotFoundError(f"Image {filename_img} not found in any chunk directory under {self.config.img_dir}")
    
    def get_batch_data(self, n_samples: int) -> Generator[Tuple[List[Image.Image], List[str]], None, None]:
        for idx_start in range(0, n_samples, self.config.batch_size):
            idx_end = min(idx_start + self.config.batch_size, n_samples)
            df_batch = self.df_meta[idx_start:idx_end]
            
            imgs = [self.load_img(row["fileNameAsDelivered"]) for _, row in df_batch.iterrows()]
            labels_targ = df_batch["scientificName"].tolist()
            
            yield imgs, labels_targ
    
    def get_dataloaders(self):
        return None, self 