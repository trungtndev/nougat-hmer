"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import argparse
import datetime
import os
from os.path import basename
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    GradientAccumulationScheduler,
)
from sconf import Config

from nougat import NougatDataset
from lightning_module import NougatDataPLModule, NougatModelPLModule
from nougat.utils.dataset import CROHMEDataset
from nougat.utils import *

try:
    import wandb
    from pytorch_lightning.loggers import WandbLogger as Logger
except ModuleNotFoundError:
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger as Logger

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(config):
    """
    Train a Nougat model using the provided configuration.

    Args:
        `config` (dict): A dictionary containing configuration settings for training.
    """
    pl.seed_everything(config.get("seed", 42), workers=True)

    model_module = NougatModelPLModule(config)
    data_module = NougatDataPLModule(config)

    train_datasets = CROHMEDataset(
        config.dataset_paths+"/train",
        nougat_model=model_module.model,
        max_length=config.max_length,
        split="train",
    )
    val_datasets = CROHMEDataset(
        config.dataset_paths+"/val",
        nougat_model=model_module.model,
        max_length=config.max_length,
        split="validation",
    )
    data_module.train_datasets = train_datasets
    data_module.val_datasets = val_datasets

    lr_callback = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        dirpath=Path(config.result_path) / config.exp_name / config.exp_version,
    )
    # grad_norm_callback = GradNormCallback()
    # custom_ckpt = CustomCheckpointIO()

    # logger = Logger(config.exp_name, project="Nougat", config=dict(config))

    trainer = pl.Trainer(
        num_nodes=config.get("num_nodes", 1),
        devices="auto",
        strategy="ddp_find_unused_parameters_true",
        accelerator="auto",
        # plugins=[SLURMEnvironment(auto_requeue=False)],
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        # limit_val_batches=config.val_batches,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=15,
        precision="16",
        num_sanity_val_steps=0,
        # logger=logger,
        callbacks=[
            # lr_callback,
            # grad_norm_callback,
            # checkpoint_callback,
            # GradientAccumulationScheduler({0: config.accumulate_grad_batches}),
        ],
    )

    trainer.fit(
        model_module,
        data_module,
        ckpt_path=config.get("resume_from_checkpoint_path", None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_version", type=str, required=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--job", type=int, default=None)
    args, left_argv = parser.parse_known_args()

    config = Config(args.config)
    config.argv_update(left_argv)
    config.debug = args.debug
    config.job = args.job
    if not config.get("exp_name", False):
        config.exp_name = basename(args.config).split(".")[0]
    config.exp_version = (
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.exp_version
        else args.exp_version
    )

    save_config_file(
        config, Path(config.result_path) / config.exp_name / config.exp_version
    )
    train(config)
