import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import numpy as np
from pathlib import Path
import glob
from utils.common import set_seed
from embeddings import EmbeddingStorage, generate_embeddings_batched


def find_most_recent_embedding_file(storage_dir: str) -> str:
    """Find the most recently created embedding file in storage directory."""
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        raise FileNotFoundError(f"Storage directory {storage_dir} does not exist")
    
    # Find all .h5 files in storage directory
    h5_files = list(storage_path.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {storage_dir}")
    
    # Sort by modification time (most recent first)
    h5_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    most_recent = h5_files[0]
    print(f"Found {len(h5_files)} embedding file(s), using most recent: {most_recent.name}")
    return str(most_recent)


@hydra.main(config_path="conf", config_name="experiment", version_base="1.1")
def main(cfg: DictConfig) -> None:

    set_seed(cfg.experiment.seed)
    print(f"Starting {cfg.experiment.name}")
    print(f"Mode: {cfg.mode}")
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if cfg.mode == "embed":
        # EMBED MODE: Generate and save embeddings
        print("\n" + "="*60)
        print("EMBEDDING GENERATION MODE")
        print("="*60)
        
        # Initialize model and data loader
        model = instantiate(cfg.model)
        print(f"Model info: {model.get_model_info()}")

        data_loader = instantiate(cfg.data)
        print(f"Dataset info: {data_loader.get_dataset_info()}")
        
        # Generate embeddings in batches
        embeddings, metadata, generated_filepath = generate_embeddings_batched(
            model=model, 
            data_loader=data_loader, 
            device=cfg.experiment.device,
            max_samples=cfg.embeddings.max_samples,
            auto_filename=cfg.embeddings.auto_filename,
            base_storage_dir=cfg.embeddings.storage_dir
        )
        
        # Determine final file path
        if cfg.embeddings.custom_filename:
            final_filepath = cfg.embeddings.custom_filename
        else:
            final_filepath = generated_filepath
        
        # Save embeddings to HDF5
        storage = EmbeddingStorage(final_filepath)
        storage.save_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            model_name=model.model_name,
            dataset_info=data_loader.get_dataset_info(),
            overwrite=True
        )
        
        print(f"Embeddings saved to: {final_filepath}")
        print("Run with mode='eval' to evaluate using these embeddings")

    elif cfg.mode == "eval":
        # EVAL MODE: Load embeddings and run evaluation
        print("\n" + "="*60)
        print("EVALUATION MODE (with pre-computed embeddings)")
        print("="*60)
        
        # Determine which embedding file to use
        if cfg.evaluation.embedding_file:
            embedding_file_path = cfg.evaluation.embedding_file
            print(f"Using specified embedding file: {embedding_file_path}")
        else:
            embedding_file_path = find_most_recent_embedding_file(cfg.embeddings.storage_dir)
        
        # Load embeddings from HDF5
        storage = EmbeddingStorage(embedding_file_path)
        embeddings, metadata_df, attributes = storage.load_embeddings()
        
        print(f"Loaded embeddings generated with model: {attributes.get('model_name', 'Unknown')}")
        print(f"Generated on: {attributes.get('creation_timestamp', 'Unknown')}")
        
        # Split into query and database sets
        print("\nSplitting data into query and database sets...")
        np.random.seed(cfg.experiment.seed)
        n_total = len(embeddings)
        query_size = max(1, int(n_total * cfg.evaluation.split.query_ratio))
        
        # Random split
        indices = np.random.permutation(n_total)
        query_indices = indices[:query_size]
        database_indices = indices[query_size:]
        
        query_embeddings = embeddings[query_indices]
        query_metadata = metadata_df.iloc[query_indices].reset_index(drop=True)
        database_embeddings = embeddings[database_indices]
        database_metadata = metadata_df.iloc[database_indices].reset_index(drop=True)
        
        print(f"Query set size: {len(query_embeddings)}")
        print(f"Database set size: {len(database_embeddings)}")
        
        # Initialize model (needed for evaluator interface compatibility)
        model = instantiate(cfg.model)
        print(f"Model info: {model.get_model_info()}")

        data_loader = instantiate(cfg.data)
        print(f"Dataset info: {data_loader.get_dataset_info()}")
        
        # Run evaluators
        results = {}
        for i, ev_cfg in enumerate(cfg.evaluators, 1):
            evaluator = instantiate(ev_cfg)
            print(f"\nEvaluation info: {evaluator.get_evaluator_info()}")
            
            # Pass embeddings and metadata to evaluator
            loss, metric = evaluator.evaluate_with_embeddings(
                query_embeddings=query_embeddings,
                query_metadata=query_metadata,
                database_embeddings=database_embeddings,
                database_metadata=database_metadata,
                device=cfg.experiment.device
            )
            results[evaluator.evaluator_name] = {"loss": loss, "metric": metric}
              
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        for eval_name, result in results.items():
            print(f"{eval_name:25}: metric={result['metric']:.4f}")
    
    elif cfg.mode == "train":
        print("Training mode not implemented yet")
    
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Supported modes: embed, eval, train")

    print(f"Completed {cfg.experiment.name}")


if __name__ == "__main__":
    main() 