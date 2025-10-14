from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import torch
import os

from src.transforms import build_transforms
from src.pandas_data import WaferMapDataset
from src.models import build_model

def eval_main(cfg):
    test_df = pd.read_pickle(cfg["test_data_path"])
    classes = sorted(test_df["failureType"].unique())
    num_classes = len(classes)
    train_tf, eval_tf = build_transforms(cfg["img_size"])
    test_imgs = test_df["waferMap"].values
    test_lbls = test_df["label_idx"].values

    test_ds = WaferMapDataset(test_imgs, test_lbls, transform=eval_tf)

    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = cfg["device"] if cfg["device"]!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes).to(device)

    model.load_state_dict(torch.load(os.path.join(cfg['out_dir'], "best_model.pt"), map_location=device))
    model.eval()
    y_true_test, y_pred_test = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.float().to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true_test.extend(labels.cpu().numpy())
            y_pred_test.extend(preds.cpu().numpy())

    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_f1 = f1_score(y_true_test, y_pred_test, average="macro")
    print(f"Test Accuracy: {test_acc:.4f} | Test Macro F1: {test_f1:.4f}\n")
    report = classification_report(y_true_test, y_pred_test, output_dict=True)
    print(report)

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(cfg['out_dir'], "classification_report.csv"))

    return test_f1

if __name__ == "__main__":
    import yaml
    cfg_file = "configs/config.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    eval_main(cfg)