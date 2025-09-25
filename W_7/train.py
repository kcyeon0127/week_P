
"""Train BERT classifier on KLUE YNAT with PyTorch Lightning."""
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import YNATDataModule
from models import LightningBertClassifier


def main():
    # 기본 학습 설정 (필요 시 아래 값을 수정하세요)
    model_name = "bert-base-multilingual-cased"
    learning_rate = 2e-5
    batch_size = 16
    max_length = 128
    epochs = 3
    seed = 42
    num_workers = 4
    devices = 1  # 예: 1 또는 [0, 1]
    output_dir = Path("outputs")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pl.seed_everything(seed, workers=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    model = LightningBertClassifier(model_name=model_name, lr=learning_rate)
    datamodule = YNATDataModule(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        num_workers=num_workers,
        seed=seed,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices if accelerator == "gpu" else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(
                dirpath=str(output_dir),
                filename="bert-ynat-{epoch:02d}-{val_f1:.3f}",
                monitor="val/f1",
                mode="max",
                save_top_k=1,
            )
        ],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model=trainer.model, datamodule=datamodule)
    print(f"Best checkpoint saved to: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
