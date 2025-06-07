import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from utils.common import set_seed


@hydra.main(config_path="conf", config_name="experiment", version_base="1.1")
def main(cfg: DictConfig) -> None:

    set_seed(cfg.experiment.seed)
    print(f"Starting {cfg.experiment.name}")
    print(f"Mode: {cfg.mode}")
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = instantiate(cfg.model)
    print(f"Model info: {model.get_model_info()}")

    data_loader = instantiate(cfg.data)
    print(f"Dataset info: {data_loader.get_dataset_info()}")

    if cfg.mode == "eval":
        results = {}
        for i, ev_cfg in enumerate(cfg.evaluators, 1):
            evaluator = instantiate(ev_cfg)
            print(f"Evaluation info: {evaluator.get_evaluator_info()}")
            
            loss, metric = evaluator.evaluate(model, data_loader, cfg.experiment.device)
            results[evaluator.evaluator_name] = {"loss": loss, "metric": metric}
              
        for eval_name, result in results.items():
            print(f"{eval_name:25}: loss={result['loss']:.4f}, metric={result['metric']:.4f}")
    
    elif cfg.mode == "train":
        print("Training mode not implemented yet")
    
    print(f"Completed {cfg.experiment.name}")


if __name__ == "__main__":
    main() 