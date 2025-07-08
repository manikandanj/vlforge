import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Any
from .base import BaseDataLoader


class VLMBioDataLoader(BaseDataLoader):
    
    def __init__(self, base_dir: str, group: str, batch_size: int = 32, 
                 metadata_file: Optional[str] = None, **kwargs):
        self.base_dir = Path(base_dir).resolve()
        self.group = group
        self.metadata_file = metadata_file or "metadata_easy.csv"  # Default to easy dataset
        super().__init__(batch_size=batch_size, **kwargs)
    
    def _setup_data(self):
        self.group_dir = self.base_dir / self.group
        self.metadata_dir = self.group_dir / "metadata"
        self.metadata_path = self.metadata_dir / self.metadata_file
        self.img_dir = self.group_dir
        # Load metadata
        self.df_meta = pd.read_csv(self.metadata_path)
        self.chunk_dirs = list(self.group_dir.glob("chunk_*"))
        self._labels = sorted(self.df_meta["scientificName"].dropna().unique())


    def load_image(self, image_identifier: str) -> Image.Image:
        for chunk_dir in self.chunk_dirs:
            img_path = chunk_dir / image_identifier
            if img_path.exists():
                return Image.open(img_path).convert("RGB")
    
    def get_labels(self) -> List[str]:
        return self._labels
    
    def get_metadata_for_ids(self, unique_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific fileNameAsDelivered values"""
        metadata_list = []
        
        for unique_id in unique_ids:
            # Find the row with matching fileNameAsDelivered
            matching_rows = self.df_meta[self.df_meta["fileNameAsDelivered"] == unique_id]
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]  # Take first match
                metadata = {
                    "scientificName": row.get("scientificName", ""),
                    "fileNameAsDelivered": row.get("fileNameAsDelivered", ""),
                    # Add any other fields that might be useful
                }
                # Add any additional fields that exist in the dataframe
                for col in self.df_meta.columns:
                    if col not in metadata:
                        metadata[col] = row.get(col, "")
            else:
                # Return empty metadata if not found
                metadata = {
                    "scientificName": "",
                    "fileNameAsDelivered": unique_id,
                }
                
            metadata_list.append(metadata)
            
        return metadata_list
    
    
    def get_batch_data(self, n_samples: int) -> Generator[Tuple[List[Image.Image], List[str], List[str]], None, None]:
        for idx_start in range(0, n_samples, self.batch_size):
            idx_end = min(idx_start + self.batch_size, n_samples)
            df_batch = self.df_meta[idx_start:idx_end]
            
            images = []
            labels = []
            unique_ids = []
            
            for _, row in df_batch.iterrows():
                try:
                    img = self.load_image(row["fileNameAsDelivered"])
                    images.append(img)
                    labels.append(row["scientificName"])
                    unique_ids.append(row["fileNameAsDelivered"])
                except FileNotFoundError as e:
                    print(f"Warning: {e}")
                    continue
            
            if images: 
                yield images, labels, unique_ids 
