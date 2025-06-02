import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from utils.common import set_seed

@hydra.main(config_path="conf", config_name="experiment", version_base="1.1")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.experiment.seed)
    
    print(f"Device: {cfg.experiment.device}")
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else "cpu")
    
    model = instantiate(cfg.model)
    data_obj = instantiate(cfg.data)
    _, val_loader = data_obj.get_dataloaders()
    
    for ev_cfg in cfg.evaluators:
        evaluator = instantiate(ev_cfg)
        loss, acc = evaluator.evaluate(model, val_loader, cfg.experiment.device)
        print(f"{evaluator.__class__.__name__}: loss={loss:.4f}, acc={acc:.4f}")

if __name__ == "__main__":
    main() 