import os
import torch


class CheckpointManager:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.best_metric = self._load_best_metric()

    def _load_best_metric(self):
        path = os.path.join(self.out_dir, "best_metric.txt")
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    return float(f.read().strip())
            except Exception:
                pass
        return -float("inf")

    def _save_best_metric(self, metric_value):
        path = os.path.join(self.out_dir, "best_metric.txt")
        with open(path, "w") as f:
            f.write(str(metric_value))

    def save(self, epoch, model, metric_value):
        path = os.path.join(self.out_dir, "best_model.pt")
        torch.save(model.state_dict(), path)
        self._save_best_metric(metric_value)
        return path

    def step(self, epoch, model, metric_value):
        if metric_value > self.best_metric:
            self.save(epoch, model, metric_value)
            self.best_metric = metric_value
            return True
        return False

