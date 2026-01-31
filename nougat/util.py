from typing import List

from pytorch_lightning.plugins import CheckpointIO
from pytorch_lightning.utilities import rank_zero_only
from pathlib import Path
import torch
from pytorch_lightning.callbacks import (
    Callback,
)
import os
from torchmetrics import Metric


class CustomCheckpointIO(CheckpointIO):
    """
    A custom class for saving and loading checkpoints with additional functionality.

    Args:
        `CheckpointIO` (class): The base class for checkpoint I/O operations.

    Methods:
        `save_checkpoint(checkpoint, path, storage_options=None)`:
            Save a checkpoint to the specified path.

        `load_checkpoint(path, storage_options=None)`:
            Load a checkpoint from the specified path.

        `remove_checkpoint(path) -> None`:
            Remove a checkpoint from the specified path.

    """

    @rank_zero_only
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        """
        Save a checkpoint to the specified path.

        Args:
            `checkpoint` (dict): The dictionary containing the checkpoint data.
            `path` (str): The path where the checkpoint will be saved.
            `storage_options` (dict, optional): Additional storage options.
        """
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        """
        Load a checkpoint from the specified path.

        Args:
            `path` (str): The path from which the checkpoint will be loaded.
            `storage_options` (dict, optional): Additional storage options.
        """
        path = Path(path)
        if path.is_file():
            print("path:", path, path.is_dir())
            ckpt = torch.load(path)
            if not "state_dict" in ckpt:
                ckpt["state_dict"] = {
                    "model." + key: value
                    for key, value in torch.load(
                        path.parent / "pytorch_model.bin"
                    ).items()
                }
            return ckpt
        else:
            checkpoint = torch.load(path / "artifacts.ckpt")
            state_dict = torch.load(path / "pytorch_model.bin")
            checkpoint["state_dict"] = {
                "model." + key: value for key, value in state_dict.items()
            }
            return checkpoint

    def remove_checkpoint(self, path) -> None:
        return super().remove_checkpoint(path)


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    @staticmethod
    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("train/grad_norm", self.gradient_norm(model))


@rank_zero_only
def save_config_file(config, path):
    if not Path(path).exists():
        os.makedirs(path)
    save_path = Path(path) / "config.yaml"
    print(config.dumps())
    with open(save_path, "w") as f:
        f.write(config.dumps(modified_color=None, quote_str=True))
        print(f"Config is saved at {save_path}")

class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[str], indices: List[str]):
        for pred, truth in zip(indices_hat, indices):
            is_same = pred == truth
            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate