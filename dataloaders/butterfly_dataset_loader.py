import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Generator, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import functools
from .base import BaseDataLoader
from utils.logging_config import get_logger


class ButterflyDatasetDataLoader(BaseDataLoader):
    
    def __init__(self, base_dir: str, metadata_dir: str, group: str, batch_size: int = 32, 
                 metadata_file: Optional[str] = None, max_workers: int = 4, **kwargs):
        self.base_dir = Path(base_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.group = group
        self.metadata_file = metadata_file or "data_meta-nymphalidae_whole_specimen-v240606.csv"  # Default to subset
        self.max_workers = max_workers  # For parallel image loading
        super().__init__(batch_size=batch_size, **kwargs)
    
    def _setup_data(self):
        logger = get_logger()
        logger.info("Setting up optimized butterfly dataset data loader...")
        
        self.group_dir = self.base_dir / self.group
        self.metadata_path = self.metadata_dir / self.metadata_file
        self.img_dir = self.group_dir / "images"
        
        logger.info(f"Metadata path: {self.metadata_path}")
        logger.info(f"Image directory: {self.img_dir}")
        
        # Check if metadata file exists
        if not self.metadata_path.exists():
            logger.error(f"Metadata file does not exist: {self.metadata_path}")
            raise FileNotFoundError(f"Metadata file does not exist: {self.metadata_path}")
        
        # Load metadata
        logger.info("Loading metadata CSV file...")
        self.df_meta = pd.read_csv(self.metadata_path)
        logger.info(f"Successfully loaded metadata: {len(self.df_meta)} rows")
        logger.info(f"Available columns: {list(self.df_meta.columns)}")
        
        # Pre-filter valid images to avoid loading errors during training
        logger.info("Pre-validating image paths...")
        valid_rows = []
        missing_count = 0
        
        for idx, row in self.df_meta.iterrows():
            img_path = self.img_dir / row["species"] / row["mask_name"]
            if img_path.exists():
                valid_rows.append(row)
            else:
                missing_count += 1
                if missing_count <= 10:  # Log first 10 missing files
                    logger.warning(f"Missing image: {img_path}")
                    
        if missing_count > 10:
            logger.warning(f"... and {missing_count - 10} more missing images")
            
        self.df_meta = pd.DataFrame(valid_rows).reset_index(drop=True)
        logger.info(f"After validation: {len(self.df_meta)} valid images ({missing_count} missing)")
        
        self._labels = sorted(self.df_meta["species"].dropna().unique())
        logger.info(f"Found {len(self._labels)} unique species")
        
        # Calculate frequency maps
        logger.info("Calculating frequency maps...")
        self._calculate_frequency_maps()
        logger.info("Optimized dataset setup completed successfully")

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

    def _load_single_image(self, image_path: Path) -> Optional[Image.Image]:
        """Optimized single image loading with better error handling"""
        try:
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                # Pre-validate that image can be processed
                img.load()  # Force loading to catch corrupt images early
                return img
        except Exception as e:
            # Silently handle corrupted images
            pass
        return None

    def load_image(self, image_identifier: str) -> Optional[Image.Image]:
        img_path = self.img_dir / image_identifier
        return self._load_single_image(img_path)
    
    def _load_batch_images_parallel(self, image_paths: List[Path]) -> List[Optional[Image.Image]]:
        """Load multiple images in parallel for better I/O performance"""
        if self.max_workers <= 1:
            # Fallback to sequential loading
            return [self._load_single_image(path) for path in image_paths]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Use map for efficient parallel loading
            images = list(executor.map(self._load_single_image, image_paths))
        return images
    
    def get_labels(self) -> List[str]:
        return self._labels
    
    def get_metadata_for_ids(self, unique_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific mask_names - optimized with vectorized operations"""
        if not unique_ids:
            return []
        
        # Use vectorized pandas operations for better performance
        mask_name_set = set(unique_ids)
        matching_df = self.df_meta[self.df_meta["mask_name"].isin(mask_name_set)]
        
        # Create a lookup dictionary for O(1) access
        metadata_lookup = {}
        for _, row in matching_df.iterrows():
            mask_name = row["mask_name"]
            metadata = {
                "species": row.get("species", ""),
                "genus": row.get("genus", ""),
                "subfamily": row.get("subfamily", ""),
                "mask_name": mask_name,
            }
            
            # Add any additional fields that exist in the dataframe
            for col in self.df_meta.columns:
                if col not in metadata:
                    metadata[col] = row.get(col, "")
            
            metadata_lookup[mask_name] = metadata
        
        # Build result list in the same order as unique_ids
        metadata_list = []
        for unique_id in unique_ids:
            if unique_id in metadata_lookup:
                metadata_list.append(metadata_lookup[unique_id])
            else:
                # Return minimal metadata if not found
                metadata_list.append({
                    "species": "",
                    "genus": "",
                    "subfamily": "",
                    "mask_name": unique_id,
                })
                
        return metadata_list
    
    def get_batch_data(self, n_samples: Optional[int] = None) -> Generator[Tuple[List[Image.Image], List[str], List[str]], None, None]:
        # If n_samples is None, process all available samples
        if n_samples is None:
            n_samples = len(self.df_meta)
        
        logger = get_logger()
        total_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx, idx_start in enumerate(range(0, n_samples, self.batch_size)):
            idx_end = min(idx_start + self.batch_size, n_samples)
            df_batch = self.df_meta[idx_start:idx_end]
            
            # Prepare image paths for parallel loading
            image_paths = []
            expected_labels = []
            expected_unique_ids = []
            
            for _, row in df_batch.iterrows():
                img_path = self.img_dir / row["species"] / row["mask_name"]
                image_paths.append(img_path)
                expected_labels.append(row["species"])
                expected_unique_ids.append(row["mask_name"])
            
            # Load images in parallel
            loaded_images = self._load_batch_images_parallel(image_paths)
            
            # Filter out None images and corresponding metadata
            valid_images = []
            valid_labels = []
            valid_unique_ids = []
            
            for img, label, unique_id in zip(loaded_images, expected_labels, expected_unique_ids):
                if img is not None:
                    valid_images.append(img)
                    valid_labels.append(label)
                    valid_unique_ids.append(unique_id)
            
            if valid_images:
                # Log progress for optimization monitoring
                if batch_idx % 50 == 0 or batch_idx < 3:
                    success_rate = len(valid_images) / len(loaded_images) * 100
                    logger.info(f"Batch {batch_idx + 1}/{total_batches}: loaded {len(valid_images)}/{len(loaded_images)} images ({success_rate:.1f}% success)")
                
                yield valid_images, valid_labels, valid_unique_ids 
