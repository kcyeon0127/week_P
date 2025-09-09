# pip install -r requirements.txt

# # 1) 학습
# python train_ynat.py \
#   --model_name bert-base-multilingual-cased \
#   --batch_size 32 --max_length 128 --epochs 3

# # 2) 리포트 생성
# python predict_and_report.py \
#   --checkpoint outputs/last.ckpt \
#   --out_dir outputs

import argparse, os, torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from datamodule import YNATDataModule
from models import LightningBertClassifier

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--out_dir', type=str, default='outputs')

    # wandb 옵션
    p.add_argument('--wandb_project', type=str, default='week5-ynat', help='W&B project name')
    p.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (team/user). None이면 현재 계정')
    p.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    p.add_argument('--wandb_offline', action='store_true', help='Set WANDB_MODE=offline')
    p.add_argument('--wandb_log_model', type=str, default='true', help="true/false/all (체크포인트 아티팩트 업로드)")
    return p.parse_args()

def main():
    args = parse()
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'

    pl.seed_everything(args.seed, workers=True)
    dm = YNATDataModule(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed
    )

    model = LightningBertClassifier(
        model_name=args.model_name,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 임베딩 일치 여부 출력
    equal = model.backbone.embedding_equal_to_hf(args.model_name)
    print(f"[Check] Embedding equal to HF pretrained: {equal}")

    # --- W&B Logger ---
    log_model_flag = True if args.wandb_log_model.lower() in ['1','true','yes','all'] else False
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        save_dir=args.out_dir,
        log_model=log_model_flag  # 체크포인트를 W&B artifact로 업로드
    )
    # 하이퍼파라미터 기록
    wandb_logger.experiment.config.update(vars(args))
    # 모델/그래디언트 감시(선택)
    wandb_logger.watch(model, log='all', log_freq=100)

    ckpt = ModelCheckpoint(
        dirpath=args.out_dir, filename='best',
        save_top_k=1, monitor='val/f1', mode='max'
    )
    lrmon = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[ckpt, lrmon],
        precision='16-mixed' if torch.cuda.is_available() else 32,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path=ckpt.best_model_path if ckpt.best_model_path else None)

    # 마지막 체크포인트도 저장 (W&B artifact와 별개로 로컬에 보관)
    os.makedirs(args.out_dir, exist_ok=True)
    last_path = os.path.join(args.out_dir, 'last.ckpt')
    trainer.save_checkpoint(last_path)
    print(f"Saved: {last_path}")
    print(f"Best: {ckpt.best_model_path}")

if __name__ == '__main__':
    main()
