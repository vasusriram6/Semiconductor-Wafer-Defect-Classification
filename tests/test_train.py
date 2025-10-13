import yaml
from src.train import train_main

def test_train_main():
    cfg_file = "configs/config_test.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    train_f1 = train_main(cfg)
    assert isinstance(train_f1, float)