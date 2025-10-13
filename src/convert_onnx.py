import os
import torch

from src.models import build_model


def convert_onnx(torch_model_path, output_path):
    device = cfg["device"] if cfg["device"]!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(9)

    model.load_state_dict(torch.load(torch_model_path))
    model.eval()

    x = torch.randn(1, 3, cfg['img_size'], cfg['img_size'], requires_grad=True)

    torch.onnx.export(model,                     # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  output_path,              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=12,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'])    # the model's output names
    
    print("Exported model.")

if __name__ == "__main__":
    import yaml
    cfg_file = "configs/config.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    torch_model_path = os.path.join(cfg["out_dir"], "best_model.pt")
    output_path = os.path.join(cfg["out_dir"], "wafer.onnx")
    
    convert_onnx(torch_model_path, output_path)