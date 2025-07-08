from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Optional, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict, Counter


class BaseDataLoader(ABC):
    """Abstract base class for data loaders providing a unified interface"""
    
    def __init__(self, batch_size: int = 32, **kwargs):
        self.batch_size = batch_size
        self.config = kwargs
        self._setup_data()
        
    @abstractmethod
    def _setup_data(self):
        pass
    
    @abstractmethod
    def load_image(self, image_identifier: str) -> Optional[Image.Image]:
        pass
    
    @abstractmethod
    def get_labels(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_batch_data(self, n_samples: int) -> Generator[Tuple[List[Image.Image], List[str], List[str]], None, None]:
        pass

    @abstractmethod
    def get_metadata_for_ids(self, unique_ids: List[str]) -> List[Dict[str, Any]]:
        """Get metadata for specific unique IDs
        
        Args:
            unique_ids: List of unique identifiers
            
        Returns:
            List of metadata dictionaries, one per unique_id
            Each dict contains loader-specific metadata fields
        """
        pass

    def get_frequency_maps(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Get frequency maps for different taxonomic levels (if supported by the loader)
        
        Returns:
            Dictionary with frequency maps for different levels (e.g., species, genus, subfamily)
            or None if not supported by this loader
        """
        return None

    def explore_dataset(self):
        samples_by_class = defaultdict(list)
        
        for batch_images, batch_labels, batch_unique_ids in self.get_batch_data(n_samples=100):
            for img, label, unique_id in zip(batch_images, batch_labels, batch_unique_ids):
                if img is not None and len(samples_by_class[label]) < 3:
                    samples_by_class[label].append((img, label))
            break
        
        all_samples = []
        for samples in samples_by_class.values():
            all_samples.extend(samples)
        
        if all_samples:
            selected = random.sample(all_samples, min(12, len(all_samples)))
            cols = 4
            rows = (len(selected) + cols - 1) // cols
            
            plt.figure(figsize=(12, rows * 3))
            for idx, (img, label) in enumerate(selected):
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(np.array(img))
                plt.title(label)
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # for class_name, samples in samples_by_class.items():
        #     if samples:
        #         plt.figure(figsize=(9, 3))
        #         for idx, (img, label) in enumerate(samples[:3]):
        #             plt.subplot(1, 3, idx + 1)
        #             plt.imshow(np.array(img))
        #             plt.title(f"{class_name} {idx + 1}")
        #             plt.axis('off')
        #         plt.tight_layout()
        #         plt.show()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "loader_name": self.__class__.__name__,
            "batch_size": self.batch_size,
            "num_labels": len(self.get_labels()),
            "config": self.config
        }
