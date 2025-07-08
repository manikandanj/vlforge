import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import torch
import re
from utils.logging_config import get_logger


class EmbeddingStorage:
    """
    Utility class for storing and loading embeddings with metadata in HDF5 format.
    
    HDF5 Structure:
    embeddings.h5
    ├── embeddings (N, embedding_dim) - float32 array
    ├── metadata/
    │   ├── unique_ids (N,) - string array
    │   ├── species (N,) - string array  
    │   ├── genus (N,) - string array
    │   ├── subfamily (N,) - string array
    │   └── ... other metadata fields
    └── attributes/
        ├── total_samples
        ├── embedding_dim
        ├── model_name
        ├── creation_timestamp
        └── dataset_info
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        
    @staticmethod
    def generate_filename(model_name: str, dataset_info: Dict, base_dir: str = "storage") -> str:
        """
        Generate a descriptive filename using Hydra's datetime format.
        
        Format: {YYYY-MM-DD}_{HH-MM-SS}_{model_name}_{dataset_name}_embeddings.h5
        This matches Hydra's output format for easy correlation.
        
        Args:
            model_name: Name of the model (from model.model_name)
            dataset_info: Dataset information dict
            base_dir: Base directory for storage
            
        Returns:
            Full file path with descriptive name
        """
        # Get current time in Hydra format: YYYY-MM-DD/HH-MM-SS
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        
        # Clean model name (remove special chars, make filesystem-safe)
        clean_model_name = re.sub(r'[^\w\-_]', '_', model_name)
        
        # Extract dataset name from dataset_info
        dataset_name = "unknown"
        if isinstance(dataset_info, dict):
            # Try to get a meaningful dataset identifier
            loader_name = dataset_info.get('loader_name', '')
            if 'butterfly' in loader_name.lower():
                dataset_name = "butterfly"
            elif 'vlmbio' in loader_name.lower():
                dataset_name = "vlmbio"
            elif 'fish' in str(dataset_info).lower():
                dataset_name = "fish"
            else:
                # Fallback to first part of loader name
                dataset_name = loader_name.replace('DataLoader', '').lower()
        
        # Generate filename: {date}_{time}_{model}_{dataset}_embeddings.h5
        filename = f"{date_str}_{time_str}_{clean_model_name}_{dataset_name}_embeddings.h5"
        
        return str(Path(base_dir) / filename)
        
    def save_embeddings(self, embeddings: np.ndarray, metadata_list: List[Dict], 
                       model_name: str, dataset_info: Dict, overwrite: bool = False):
        """
        Save embeddings and metadata to HDF5 file.
        
        Args:
            embeddings: (N, embedding_dim) numpy array of embeddings
            metadata_list: List of metadata dictionaries for each embedding
            model_name: Name of the model used to generate embeddings
            dataset_info: Information about the dataset
            overwrite: Whether to overwrite existing file
        """
        if self.file_path.exists() and not overwrite:
            raise FileExistsError(f"File {self.file_path} already exists. Use overwrite=True to replace.")
            
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata list to structured format
        metadata_df = pd.DataFrame(metadata_list)
        
        logger = get_logger()
        logger.info(f"Saving {len(embeddings)} embeddings to {self.file_path}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
        with h5py.File(self.file_path, 'w') as f:
            # Store embeddings
            f.create_dataset('embeddings', data=embeddings.astype(np.float32), 
                           compression='gzip', compression_opts=9)
            
            # Store metadata
            metadata_group = f.create_group('metadata')
            for column in metadata_df.columns:
                values = metadata_df[column].values
                
                # Handle string data
                if values.dtype == 'object':
                    # Convert to bytes for HDF5 compatibility
                    string_values = [str(v).encode('utf-8') for v in values]
                    metadata_group.create_dataset(column, data=string_values)
                else:
                    metadata_group.create_dataset(column, data=values)
            
            # Store attributes with Hydra-compatible timestamp
            attrs_group = f.create_group('attributes')
            attrs_group.attrs['total_samples'] = len(embeddings)
            attrs_group.attrs['embedding_dim'] = embeddings.shape[1]
            attrs_group.attrs['model_name'] = model_name.encode('utf-8')
            
            # Use Hydra-compatible timestamp format
            now = datetime.now()
            hydra_timestamp = f"{now.strftime('%Y-%m-%d')}/{now.strftime('%H-%M-%S')}"
            attrs_group.attrs['creation_timestamp'] = hydra_timestamp.encode('utf-8')
            attrs_group.attrs['creation_timestamp_iso'] = now.isoformat().encode('utf-8')  # Keep ISO for programmatic use
            
            attrs_group.attrs['dataset_info'] = str(dataset_info).encode('utf-8')
            
        logger.info(f"Successfully saved embeddings to {self.file_path}")
        
    def load_embeddings(self) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """
        Load embeddings and metadata from HDF5 file.
        
        Returns:
            Tuple of (embeddings, metadata_df, attributes_dict)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Embedding file {self.file_path} not found")
            
        logger = get_logger()
        logger.info(f"Loading embeddings from {self.file_path}")
        
        with h5py.File(self.file_path, 'r') as f:
            # Load embeddings
            embeddings = f['embeddings'][:]
            
            # Load metadata
            metadata_dict = {}
            metadata_group = f['metadata']
            for key in metadata_group.keys():
                values = metadata_group[key][:]
                
                # Handle string data (decode from bytes)
                if values.dtype.char == 'S':  # String/bytes data
                    values = [v.decode('utf-8') for v in values]
                    
                metadata_dict[key] = values
                
            metadata_df = pd.DataFrame(metadata_dict)
            
            # Load attributes
            attributes = {}
            if 'attributes' in f:
                attrs_group = f['attributes']
                for key in attrs_group.attrs:
                    value = attrs_group.attrs[key]
                    # Decode bytes to string if needed
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    attributes[key] = value
            
        logger.info(f"Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
        
        return embeddings, metadata_df, attributes
    
    def get_file_info(self) -> Dict:
        """Get basic information about the embedding file without loading all data."""
        if not self.file_path.exists():
            return {"exists": False}
            
        with h5py.File(self.file_path, 'r') as f:
            info = {"exists": True}
            
            if 'embeddings' in f:
                info['embeddings_shape'] = f['embeddings'].shape
                
            if 'metadata' in f:
                info['metadata_fields'] = list(f['metadata'].keys())
                
            if 'attributes' in f:
                attrs = {}
                for key in f['attributes'].attrs:
                    value = f['attributes'].attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    attrs[key] = value
                info['attributes'] = attrs
                
        return info


def generate_embeddings_batched(model, data_loader, device: str, batch_size: int = 128, 
                               max_samples: Optional[int] = None, 
                               auto_filename: bool = True, base_storage_dir: str = "storage") -> Tuple[np.ndarray, List[Dict], str]:
    """
    Generate embeddings for all data in batches to handle memory constraints.
    
    Args:
        model: Model with get_image_embeddings method
        data_loader: Data loader that provides batched data
        device: Device to run model on
        batch_size: Batch size for processing (data_loader batch_size will be used)
        max_samples: Maximum number of samples to process (None for all)
        auto_filename: Whether to auto-generate filename based on model and dataset
        base_storage_dir: Base directory for storage files
        
    Returns:
        Tuple of (embeddings_array, metadata_list, generated_filepath)
    """
    logger = get_logger()
    logger.info("Generating embeddings in batches...")
    
    all_embeddings = []
    all_metadata = []
    total_processed = 0
    
    for batch_idx, (images, labels, unique_ids) in enumerate(data_loader.get_batch_data(n_samples=max_samples)):
        if not images:
            continue
            
        if batch_idx % 10 == 0:  # Log every 10th batch to reduce clutter
            logger.info(f"Processing batch {batch_idx + 1}: {len(images)} images")
        
        # Generate embeddings for this batch
        with torch.no_grad():
            embeddings = model.get_image_embeddings(images)
            
            # Convert to numpy and move to CPU if needed
            if torch.is_tensor(embeddings):
                embeddings_np = embeddings.cpu().numpy()
            else:
                embeddings_np = embeddings
                
        all_embeddings.append(embeddings_np)
        
        # Get metadata for this batch
        metadata_batch = data_loader.get_metadata_for_ids(unique_ids)
        all_metadata.extend(metadata_batch)
        
        total_processed += len(images)
        
        # Check if we've reached max_samples
        if max_samples and total_processed >= max_samples:
            break
    
    # Combine all embeddings
    if all_embeddings:
        combined_embeddings = np.vstack(all_embeddings)
        logger.info(f"Generated embeddings shape: {combined_embeddings.shape}")
        logger.info(f"Total processed: {total_processed} images")
        
        # Generate filename if auto_filename is enabled
        generated_filepath = ""
        if auto_filename:
            generated_filepath = EmbeddingStorage.generate_filename(
                model_name=model.model_name,
                dataset_info=data_loader.get_dataset_info(),
                base_dir=base_storage_dir
            )
        
        return combined_embeddings, all_metadata, generated_filepath
    else:
        raise ValueError("No embeddings generated - check data loader and model") 