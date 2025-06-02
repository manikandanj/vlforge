from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    base_dir: str
    group: str
    batch_size: int = 32
    num_workers: int = 4
    
    @property
    def img_dir(self) -> Path:
        return Path(self.base_dir) / self.group
    
    @property
    def metadata_path(self) -> Path:
        return Path(self.base_dir) / self.group / "metadata" / "metadata_10k.csv" 