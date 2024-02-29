
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path='conf', config_name="training_config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    if cfg.use_neptune_logger:
        print("using logger")

if __name__ == "__main__":
    my_app()
