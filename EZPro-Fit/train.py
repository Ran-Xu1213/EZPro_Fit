import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from EZPro_Fit.utils.parsing import parse_train_args
from EZPro_Fit.datasets import AFDBDataset, seq_collate
from EZPro_Fit.flow_wrapper import PMPNNWrapper

import torch
import wandb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import random_split

# Set precision
torch.set_float32_matmul_precision('medium')

# Parse arguments
args = parse_train_args()

# Initialize wandb
if args.wandb:
    wandb.init(
        entity='coarse-graining-mit',
        settings=wandb.Settings(start_method="fork"),
        project="codon",
        name=args.run_name,
        config=args,
    )

# Load dataset
full_ds = AFDBDataset(args)

# Split into train/val
train_len = int(len(full_ds) * 0.95)
if len(full_ds) < 30:
    train_ds = val_ds = full_ds
else:
    train_ds, val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])

print('train, val lens', len(train_ds), len(val_ds))

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=seq_collate,
    shuffle=True,
)

val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    collate_fn=seq_collate,
    shuffle=False,
)

# Initialize model
model = PMPNNWrapper(args)

# Overfit mode
if args.overfit:
    val_loader = train_loader

# Configure trainer
trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_epochs=args.epochs,
    limit_train_batches=args.train_batches or 1.0,
    limit_val_batches=args.val_batches or 1.0,
    num_sanity_val_steps=0,
    enable_progress_bar=True,
    gradient_clip_val=args.grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            save_top_k=1,
            save_last=True,
            every_n_epochs=args.ckpt_freq,
        ),
        ModelSummary(max_depth=2),
    ],
    accumulate_grad_batches=args.accumulate_grad,
    check_val_every_n_epoch=args.val_epoch_freq,
    val_check_interval=args.val_check_interval,
    logger=False,
)

# Set random seeds
torch.manual_seed(1)
np.random.seed(1)

# Train or validate
if args.validate:
    trainer.validate(model, val_loader, ckpt_path=args.ckpt)
else:
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
