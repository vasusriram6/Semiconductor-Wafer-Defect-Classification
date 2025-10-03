import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
import tqdm
from sklearn.utils.class_weight import compute_class_weight

from .pandas_data import load_and_process_pandas_data, WaferMapDataset#, load_and_filter_pkl, stratified_split
from .transforms import build_transforms
from .models import build_model
from .losses import FocalLoss
from .checkpoint import CheckpointManager

def train_main(cfg):
        
    
    (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls), classes = load_and_process_pandas_data(data_path=cfg["data_path"], seed=cfg["seed"])
    num_classes = len(classes)
    train_tf, eval_tf = build_transforms(cfg["img_size"])

    train_ds = WaferMapDataset(train_imgs, train_lbls, transform=train_tf)
    val_ds = WaferMapDataset(val_imgs, val_lbls, transform=eval_tf)
    test_ds = WaferMapDataset(test_imgs, test_lbls, transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = cfg["device"] if cfg["device"]!="auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)

    class_weights = None
    if cfg["use_class_weights"]:
        class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(train_lbls), y=train_lbls)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    criterion = FocalLoss(gamma=cfg["focal_gamma"], alpha=class_weights, reduction='mean') if cfg["use_focal"] else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    ckpt = CheckpointManager(cfg["out_dir"])

    class EarlyStopper:
        def __init__(self, patience=15):
            self.patience = patience
            self.best = None
            self.counter = 0
        def step(self, val_metric):
            if self.best is None or val_metric > self.best:
                self.best = val_metric
                self.counter = 0
                return False
            else:
                self.counter += 1
                return self.counter >= self.patience

    # Load model from checkpoint if it exists
    checkpoint_path = os.path.join(cfg["out_dir"], "checkpoint_last.pt")
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=cfg["device"] if cfg["device"] != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        best_val_acc = 0.0
        best_val_f1 = 0.0
        start_epoch = 0

    early = EarlyStopper()
    model.to(device)
    # for epoch in range(cfg["epochs"]):
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        losses, y_true, y_pred = [], [], []

        print(f"Device for model parameters: {next(model.parameters()).device}")

        train_loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg["epochs"]}", ncols=80)
        for x, y in train_loader:
            x, y = x.float().to(device), torch.tensor(y).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        scheduler.step()

        model.eval()
        val_losses, val_true, val_pred = [], [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.float().to(device), torch.tensor(y).to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                val_true.extend(y.cpu().numpy())
                val_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="macro")

        #ckpt.step(epoch, model, val_acc)
        ckpt.step(epoch, model, val_f1)

        print(f"Epoch {epoch+1} train_loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc  
        #     torch.save(model.state_dict(), "best_model.pt")

        # if val_f1 > best_val_f1:
        #     best_val_f1 = val_f1  
        #     torch.save(model.state_dict(), "best_model.pt")


        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,   # save the best metric
            "best_val_f1": best_val_f1
        }, checkpoint_path)

        # if early.step(val_acc):
        #     print("Early stopping.")
        #     break

        if early.step(val_f1):
            print("Early stopping.")
            break

    # final test
    model.load_state_dict(torch.load(os.path.join(cfg["out_dir"], "best_model.pt")))
    model.eval()
    test_true, test_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.float().to(device), torch.tensor(y).to(device)
            logits = model(x)
            test_true.extend(y.cpu().numpy())
            test_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average="macro")
    print(f"Test acc={test_acc:.4f}, f1={test_f1:.4f}")

    # export ONNX
    dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"], device=device)
    torch.onnx.export(model, dummy, os.path.join(cfg["out_dir"], "wafer.onnx"), opset_version=12)
    print("Exported model.")

if __name__ == "__main__":
    import yaml
    cfg_file = "configs/config.yaml"
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    train_main(cfg)
