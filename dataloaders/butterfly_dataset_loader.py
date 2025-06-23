import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict
from .base import BaseDataLoader


class ButterflyDatasetDataLoader(BaseDataLoader):
    
    def __init__(self, base_dir: str, metadata_dir: str, group: str, batch_size: int = 32, 
                 metadata_file: Optional[str] = None, **kwargs):
        self.base_dir = Path(base_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.group = group
        self.metadata_file = metadata_file or "data_meta-nymphalidae_whole_specimen-v240606.csv"  # Default to subset
        super().__init__(batch_size=batch_size, **kwargs)
    
    def _setup_data(self):
        self.group_dir = self.base_dir / self.group
        self.metadata_path = self.metadata_dir / self.metadata_file
        self.img_dir = self.group_dir / "images"
        print(f"img_dir: {self.img_dir}")
        print(f"metadata_path: {self.metadata_path}")
        # Load metadata
        self.df_meta = pd.read_csv(self.metadata_path)
        self._labels = sorted(self.df_meta["species"].dropna().unique())
        
        # Calculate frequency maps
        self._calculate_frequency_maps()

    def _calculate_frequency_maps(self):
        """Calculate frequency maps for species, genus, and subfamily"""
        self.species_counts = {}
        self.genus_counts = {}
        self.subfamily_counts = {}

        for _, row in self.df_meta.iterrows():
            species = row['species']
            genus = row['genus']
            subfamily = row['subfamily']

            self.species_counts[species] = self.species_counts.get(species, 0) + 1
            self.genus_counts[genus] = self.genus_counts.get(genus, 0) + 1
            self.subfamily_counts[subfamily] = self.subfamily_counts.get(subfamily, 0) + 1

    def get_frequency_maps(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Get frequency maps for species, genus, and subfamily"""
        return {
            "species": self.species_counts,
            "genus": self.genus_counts,
            "subfamily": self.subfamily_counts
        }

    def load_image(self, image_identifier: str) -> Optional[Image.Image]:
        img_path = self.img_dir / image_identifier
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
