import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import numpy as np
from pathlib import Path
import glob
from utils.common import set_seed
from utils.logging_config import setup_logging, get_logger
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
    logger = get_logger()
    logger.info(f"Found {len(h5_files)} embedding file(s), using most recent: {most_recent.name}")
    return str(most_recent)



@hydra.main(config_path="conf", config_name="experiment", version_base="1.1")
def main(cfg: DictConfig) -> None:

    # Set up logging with configuration from config file
    log_config = cfg.get('logging', {})
    logger = setup_logging(
        experiment_name=cfg.experiment.name,
        outputs_dir=log_config.get('outputs_dir', 'outputs'),
        log_filename=log_config.get('log_filename', None),
        auto_filename=log_config.get('auto_filename', False)
    )
    
    set_seed(cfg.experiment.seed)
    logger.info(f"Starting experiment: {cfg.experiment.name}")
    logger.info(f"Mode: {cfg.mode}")
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if cfg.mode == "embed":
        # EMBED MODE: Generate and save embeddings
        logger.info("EMBEDDING GENERATION MODE")
        
        # Initialize model and data loader
        logger.info("Initializing model...")
        model = instantiate(cfg.model)
        logger.info(f"Model: {model.get_model_info()}")

        logger.info("Initializing data loader...")
        try:
            data_loader = instantiate(cfg.data)
            logger.info(f"Dataset: {data_loader.get_dataset_info()}")
        except Exception as e:
            logger.error(f"Failed to initialize data loader: {str(e)}")
            raise
        
        # Generate embeddings in batches
        logger.info("Starting embedding generation...")
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
        logger.info("Saving embeddings to storage...")
        storage = EmbeddingStorage(final_filepath)
        storage.save_embeddings(
            embeddings=embeddings,
            metadata_list=metadata,
            model_name=model.model_name,
            dataset_info=data_loader.get_dataset_info(),
            overwrite=True
        )
        
        logger.info(f"Embeddings saved to: {final_filepath}")
        logger.info("Run with mode='eval' to evaluate using these embeddings")

    elif cfg.mode == "eval":
        # EVAL MODE: Choose between H5-based or model-based evaluation
        use_precomputed = cfg.evaluation.get('use_precomputed_embeddings', True)
        
        if use_precomputed:
            # H5-BASED EVALUATION: Load embeddings and run evaluation
            logger.info("EVALUATION MODE (H5-based with pre-computed embeddings)")
            
            # Determine which embedding file to use
            if cfg.evaluation.embedding_file:
                embedding_file_path = cfg.evaluation.embedding_file
                logger.info(f"Using specified embedding file: {embedding_file_path}")
            else:
                embedding_file_path = find_most_recent_embedding_file(cfg.embeddings.storage_dir)
            
            # Load embeddings from HDF5
            storage = EmbeddingStorage(embedding_file_path)
            embeddings, metadata_df, attributes = storage.load_embeddings()
            
            logger.info(f"Loaded embeddings from model: {attributes.get('model_name', 'Unknown')}")
            logger.info(f"Generated on: {attributes.get('creation_timestamp', 'Unknown')}")
            
            # Split into query and database sets
            logger.info("Splitting data into query and database sets...")
            # Seed already set in deterministic mode, but ensure it's applied for this operation
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
            
            # Apply num_queries limit if specified
            if hasattr(cfg.evaluation.split, 'num_queries') and cfg.evaluation.split.num_queries:
                num_queries_limit = cfg.evaluation.split.num_queries
                if len(query_embeddings) > num_queries_limit:
                    logger.info(f"Limiting evaluation to {num_queries_limit} queries (from {len(query_embeddings)} available)")
                    query_embeddings = query_embeddings[:num_queries_limit]
                    query_metadata = query_metadata.iloc[:num_queries_limit].reset_index(drop=True)
            
            logger.info(f"Query set: {len(query_embeddings)} samples")
            logger.info(f"Database set: {len(database_embeddings)} samples")
            
            # NOTE: No need to initialize model/data_loader for H5-based evaluation
            # Using precomputed embeddings, so model loading is skipped for performance
            
            # Run evaluators with pre-computed embeddings
            results = {}
            for i, ev_cfg in enumerate(cfg.evaluators, 1):
                evaluator = instantiate(ev_cfg)
                logger.info(f"Running evaluator: {evaluator.get_evaluator_info()}")
                
                # Pass embeddings and metadata to evaluator
                loss, metric = evaluator.evaluate_with_embeddings(
                    query_embeddings=query_embeddings,
                    query_metadata=query_metadata,
                    database_embeddings=database_embeddings,
                    database_metadata=database_metadata,
                    device=cfg.experiment.device
                )
                results[evaluator.evaluator_name] = {"loss": loss, "metric": metric}
                
        else:
            # MODEL-BASED EVALUATION: Process images through model in real-time
            logger.info("EVALUATION MODE (Model-based with real-time inference)")
            
            # Initialize model and data loader
            logger.info("Initializing model...")
            model = instantiate(cfg.model)
            logger.info(f"Model: {model.get_model_info()}")

            logger.info("Initializing data loader...")
            try:
                data_loader = instantiate(cfg.data)
                logger.info(f"Dataset: {data_loader.get_dataset_info()}")
            except Exception as e:
                logger.error(f"Failed to initialize data loader: {str(e)}")
                raise
            
            # Run evaluators with model and data loader
            results = {}
            for i, ev_cfg in enumerate(cfg.evaluators, 1):
                evaluator = instantiate(ev_cfg)
                logger.info(f"Running evaluator: {evaluator.get_evaluator_info()}")
                
                # Use traditional model-based evaluation
                loss, metric = evaluator.evaluate(
                    model=model,
                    data_loader=data_loader,
                    device=cfg.experiment.device
                )
                results[evaluator.evaluator_name] = {"loss": loss, "metric": metric}
              
        logger.info("EVALUATION RESULTS:")
        for eval_name, result in results.items():
            logger.info(f"  {eval_name}: {result['metric']:.4f}")
        
        # Save results to JSON file
        import json
        from datetime import datetime
        import os
        from hydra.core.hydra_config import HydraConfig
        
        results_data = {
            "experiment_name": cfg.experiment.name,
            "timestamp": datetime.now().isoformat(),
            "model_name": getattr(cfg.model, 'model_name', cfg.model._target_.split('.')[-1]),
            "dataset_name": getattr(cfg.data, 'group', cfg.data._target_.split('.')[-1]),
            "device": cfg.experiment.device,
            "seed": cfg.experiment.seed,
            "evaluation_config": {
                "use_precomputed_embeddings": use_precomputed,
                "embedding_file": cfg.evaluation.get('embedding_file') if use_precomputed else None,
                "query_ratio": cfg.evaluation.split.query_ratio if use_precomputed else None,
                "num_queries": getattr(cfg.evaluation.split, 'num_queries', None) if use_precomputed else None
            },
            "results": {}
        }
        
        for eval_name, result in results.items():
            results_data["results"][eval_name] = result['metric']
        
        # Save in Hydra output directory
        hydra_output_dir = HydraConfig.get().runtime.output_dir
        results_file = os.path.join(hydra_output_dir, "results.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
    
    elif cfg.mode == "report":
        # REPORT MODE: Generate visualization reports from evaluation outputs
        logger.info("REPORT MODE: Generating comparison visualizations")
        
        # Import report generator
        from reports.report_generator import ReportGenerator
        
        # Initialize report generator
        report_gen = ReportGenerator(cfg.report)
        
        # Generate reports from specified output directories
        report_files = report_gen.generate_comparison_report()
        
        logger.info("REPORT GENERATION COMPLETED:")
        logger.info(f"  Report directory: {report_files.get('output_dir', 'N/A')}")
        logger.info(f"  Generated files: {len(report_files.get('files', []))}")
        
        for file_type, file_path in report_files.get('files', {}).items():
            logger.info(f"    {file_type}: {file_path}")

    elif cfg.mode == "train":
        logger.warning("Training mode not implemented yet")
    
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Supported modes: embed, eval, train")

    logger.info(f"Completed experiment: {cfg.experiment.name}")
    logger.info("="*80)


if __name__ == "__main__":
    main() 