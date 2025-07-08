import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Any
from .base import BaseDataLoader
from utils.logging_config import get_logger


class ButterflyDatasetDataLoader(BaseDataLoader):
    
    def __init__(self, base_dir: str, metadata_dir: str, group: str, batch_size: int = 32, 
                 metadata_file: Optional[str] = None, **kwargs):
        self.base_dir = Path(base_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.group = group
        self.metadata_file = metadata_file or "data_meta-nymphalidae_whole_specimen-v240606.csv"  # Default to subset
        super().__init__(batch_size=batch_size, **kwargs)
    
    def _setup_data(self):
        logger = get_logger()
        logger.info("Setting up butterfly dataset data loader...")
        
        self.group_dir = self.base_dir / self.group
        self.metadata_path = self.metadata_dir / self.metadata_file
        self.img_dir = self.group_dir / "images"
        
        logger.info(f"Metadata path: {self.metadata_path}")
        
        # Check if metadata file exists
        if not self.metadata_path.exists():
            logger.error(f"Metadata file does not exist: {self.metadata_path}")
            raise FileNotFoundError(f"Metadata file does not exist: {self.metadata_path}")
        
        # Load metadata
        logger.info("Loading metadata CSV file...")
        self.df_meta = pd.read_csv(self.metadata_path)
        logger.info(f"Successfully loaded metadata: {len(self.df_meta)} rows")
        logger.info(f"Available columns: {list(self.df_meta.columns)}")
        
        self._labels = sorted(self.df_meta["species"].dropna().unique())
        logger.info(f"Found {len(self._labels)} unique species")
        
        # Calculate frequency maps
        logger.info("Calculating frequency maps...")
        self._calculate_frequency_maps()
        logger.info("Dataset setup completed successfully")

    def _calculate_frequency_maps(self):
        """Calculate frequency maps for available taxonomic levels"""
        self.species_counts = {}
        self.genus_counts = {}
        self.subfamily_counts = {}

        # Check which columns are available
        has_genus = 'genus' in self.df_meta.columns
        has_subfamily = 'subfamily' in self.df_meta.columns
        
        logger = get_logger()
        if not has_genus:
            logger.warning("'genus' column not found in metadata. Genus frequency map will be empty.")
        if not has_subfamily:
            logger.warning("'subfamily' column not found in metadata. Subfamily frequency map will be empty.")

        for _, row in self.df_meta.iterrows():
            # Species should always be available
            species = row.get('species', '')
            if species:
                self.species_counts[species] = self.species_counts.get(species, 0) + 1
            
            # Handle genus if available
            if has_genus:
                genus = row.get('genus', '')
                if genus:
                    self.genus_counts[genus] = self.genus_counts.get(genus, 0) + 1
            
            # Handle subfamily if available  
            if has_subfamily:
                subfamily = row.get('subfamily', '')
                if subfamily:
                    self.subfamily_counts[subfamily] = self.subfamily_counts.get(subfamily, 0) + 1

    def get_frequency_maps(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Get frequency maps for available taxonomic levels"""
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
    
    def get_metadata_for_ids(self, unique_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific mask_names"""
        metadata_list = []
        
        for unique_id in unique_ids:
            # Find the row with matching mask_name
            matching_rows = self.df_meta[self.df_meta["mask_name"] == unique_id]
            
            if not matching_rows.empty:
                row = matching_rows.iloc[0]  # Take first match
                
                # Start with core fields, handling missing columns gracefully
                metadata = {
                    "species": row.get("species", ""),
                    "genus": row.get("genus", ""),  # Will be empty string if column doesn't exist
                    "subfamily": row.get("subfamily", ""),  # Will be empty string if column doesn't exist
                    "mask_name": row.get("mask_name", ""),
                }
                
                # Add any additional fields that exist in the dataframe
                for col in self.df_meta.columns:
                    if col not in metadata:
                        metadata[col] = row.get(col, "")
            else:
                # Return minimal metadata if not found
                metadata = {
                    "species": "",
                    "genus": "",
                    "subfamily": "",
                    "mask_name": unique_id,
                }
                
            metadata_list.append(metadata)
            
        return metadata_list
    
    def get_batch_data(self, n_samples: Optional[int] = None) -> Generator[Tuple[List[Image.Image], List[str], List[str]], None, None]:
        # If n_samples is None, process all available samples
        if n_samples is None:
            n_samples = len(self.df_meta)
        
        for idx_start in range(0, n_samples, self.batch_size):
            idx_end = min(idx_start + self.batch_size, n_samples)
            df_batch = self.df_meta[idx_start:idx_end]
            
            images = []
            labels = []
            unique_ids = []
            
            for _, row in df_batch.iterrows():
                try:
                    img = self.load_image(row["species"] + "/" + row["mask_name"])
                    if img is not None:  # Only add non-None images
                        images.append(img)
                        labels.append(row["species"])
                        unique_ids.append(row["mask_name"])
                    # Skip missing images silently - they're tracked in dataset stats
                except Exception as e:
                    # Skip problematic images silently - errors tracked in dataset stats
                    continue
            
            if images: 
                yield images, labels, unique_ids 
