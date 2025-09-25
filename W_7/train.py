"""Train BERT classifier on KLUE YNAT with PyTorch Lightning."""
import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import YNATDataModule
from models import LightningBertClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BERT classifier on KLUE YNAT")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="사용할 GPU 장치 수 또는 ID 리스트 (예: '1' or '0,1'). 'auto' 사용 가능",
    )
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def parse_devices(raw: str):
    if raw.lower() == "auto":
        return "auto"
    if "," in raw:
        return [int(x) for x in raw.split(",") if x.strip()]
    return int(raw)


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pl.seed_everything(args.seed, workers=True)

    os.makedirs(args.output_dir, exist_ok=True)

    model = LightningBertClassifier(model_name=args.model_name, lr=args.lr)
    datamodule = YNATDataModule(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    requested_devices = parse_devices(args.devices)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if accelerator == "cpu":
        devices = 1
    else:
        devices = requested_devices
    strategy = None
    if accelerator == "gpu" and (
        (isinstance(devices, int) and devices > 1)
        or (isinstance(devices, (list, tuple)) and len(devices) > 1)
        or devices == "auto"
    ):
        strategy = "ddp"

    checkpoint_cb = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="bert-ynat-{epoch:02d}-{val_f1:.3f}",
        monitor="val/f1",
        mode="max",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        default_root_dir=args.output_dir,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
        strategy=strategy,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=trainer.model, datamodule=datamodule)
    print(f"Best checkpoint saved to: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
