import yaml
from src.eval import eval_main

def test_eval_main():
    cfg_file = "configs/config_test.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    eval_f1 = eval_main(cfg)
    assert isinstance(eval_f1, float)

# if __name__=="__main__":
#     test_eval_main()