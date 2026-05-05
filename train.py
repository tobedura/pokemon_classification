import os
import sys
import yaml
import torch

from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train
from src.evaluate import evaluate

DATA_DIR = "data/PokemonData"
CONFIGS = [
    "configs/exp1.yaml",
    "configs/exp2.yaml",
    "configs/exp3.yaml",
    "configs/exp4.yaml",
]


def run_experiment(config_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_name = config["exp_name"]
    print(f"\n{'='*50}")
    print(f"실험 시작: {exp_name}")
    print(f"{'='*50}")

    train_loader, val_loader, classes = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        val_ratio=config["val_ratio"],
    )

    model = get_model(
        backbone=config["backbone"],
        num_classes=len(classes),
        pretrained=config["pretrained"],
        finetune=config["finetune"],
    )

    train(model, train_loader, val_loader, config, device, exp_name)

    model.load_state_dict(torch.load(f"models/{exp_name}_best.pth", map_location=device))
    evaluate(model, val_loader, classes, device, exp_name)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"사용 디바이스: {device}")

    if len(sys.argv) > 1:
        target = sys.argv[1]
        configs = [c for c in CONFIGS if target in c]
    else:
        configs = CONFIGS

    for config_path in configs:
        run_experiment(config_path, device)
