import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator, Optional
from .base import BaseDataLoader


class ButterflyDatasetDataLoader(BaseDataLoader):
    
    def __init__(self, base_dir: str, group: str, batch_size: int = 32, 
                 metadata_file: Optional[str] = None, **kwargs):
        self.base_dir = Path(base_dir).resolve()
        self.group = group
        self.metadata_file = metadata_file or "data_meta-nymphalidae_whole_specimen-v240606_subset.csv"  # Default to subset
        super().__init__(batch_size=batch_size, **kwargs)
    
    def _setup_data(self):
        self.group_dir = self.base_dir / self.group
        self.metadata_dir = self.group_dir / "metadata"
        self.metadata_path = self.metadata_dir / self.metadata_file
        self.img_dir = self.group_dir / "images"
        # Load metadata
        self.df_meta = pd.read_csv(self.metadata_path)
        self._labels = sorted(self.df_meta["species"].dropna().unique())


    def load_image(self, image_identifier: str) -> Optional[Image.Image]:
        img_path = self.img_dir / image_identifier
        print(f"img_path: {img_path}")
        if img_path.exists():
            return Image.open(img_path).convert("RGB")
        return None
    
    def get_labels(self) -> List[str]:
        return self._labels
    
    
    def get_batch_data(self, n_samples: int) -> Generator[Tuple[List[Image.Image], List[str]], None, None]:
        for idx_start in range(0, n_samples, self.batch_size):
            idx_end = min(idx_start + self.batch_size, n_samples)
            df_batch = self.df_meta[idx_start:idx_end]
            
            images = []
            labels = []
            
            for _, row in df_batch.iterrows():
                try:
                    img = self.load_image(row["species"] + "/" + row["mask_name"])
                    if img is not None:  # Only add non-None images
                        images.append(img)
                        labels.append(row["species"])
                    else:
                        print(f"Warning: Image not found: {row['mask_name']}")
                except Exception as e:
                    print(f"Warning: Error loading image {row['mask_name']}: {e}")
                    continue
            
            if images: 
                yield images, labels 
