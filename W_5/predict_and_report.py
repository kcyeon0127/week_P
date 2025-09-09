import argparse, os, torch, pandas as pd, numpy as np
import pytorch_lightning as pl
from datamodule import YNATDataModule, YNAT_LABELS
from models import LightningBertClassifier
from sklearn.metrics import confusion_matrix, f1_score
import plotly.graph_objects as go
import glob

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=False)
    p.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='outputs')
    p.add_argument('--save_png', action='store_true')
    return p.parse_args()

def main():
    args = parse()
    
    if not args.checkpoint:
        cand = [
            os.path.join(args.out_dir, 'best.ckpt'),
            os.path.join(args.out_dir, 'last.ckpt'),
        ]
        ckpts = [c for c in cand if os.path.isfile(c)]
        if not ckpts:
            any_ckpt = sorted(glob.glob(os.path.join(args.out_dir, '*.ckpt')))
            if any_ckpt:
                ckpts = [any_ckpt[-1]]  # 가장 최근 파일
        if not ckpts:
            raise FileNotFoundError(f'No checkpoint found in {args.out_dir}. '
                                    f'Please pass --checkpoint <path>')
        args.checkpoint = ckpts[0]
        print(f'[Auto] Using checkpoint: {args.checkpoint}')
    
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.out_dir, exist_ok=True)
    dm = YNATDataModule(model_name=args.model_name, batch_size=args.batch_size, max_length=args.max_length, num_workers=args.num_workers, seed=args.seed)
    dm.prepare_data(); dm.setup('test')

    model = LightningBertClassifier.load_from_checkpoint(args.checkpoint, map_location='cpu')
    model.eval()

    texts, refs, preds = [], [], []
    with torch.no_grad():
        for batch in dm.test_dataloader():
            logits = model(**{k: v for k, v in batch.items() if k in ['input_ids','attention_mask','token_type_ids']})['logits']
            pred = logits.argmax(dim=-1).cpu().numpy().tolist()
            refs += batch['labels'].cpu().numpy().tolist()
            preds += pred
            texts += batch['text']

    df = pd.DataFrame({
        'text': texts,
        'reference_label': [YNAT_LABELS[i] for i in refs],
        'predicted_label': [YNAT_LABELS[i] for i in preds],
    })
    df['correct'] = (df['reference_label'] == df['predicted_label']).astype(int)
    csv_path = os.path.join(args.out_dir, 'preds.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Saved CSV: {csv_path}")

    # Macro-F1
    macro_f1 = f1_score(refs, preds, average='macro')
    print(f"Macro-F1 (test): {macro_f1:.4f}")

    # Confusion Matrix (Plotly)
    cm = confusion_matrix(refs, preds, labels=list(range(len(YNAT_LABELS))))
    z = cm.astype(int)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=YNAT_LABELS, y=YNAT_LABELS, hoverongaps=False, zauto=True, colorbar=dict(title='Count')
    ))
    fig.update_layout(title=f'KLUE-YNAT Confusion Matrix (Macro-F1={macro_f1:.4f})', xaxis_title='Predicted', yaxis_title='True')
    html_path = os.path.join(args.out_dir, 'confusion_matrix.html')
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Saved HTML: {html_path}")
    if args.save_png:
        try:
            fig.write_image(os.path.join(args.out_dir, 'confusion_matrix.png'), scale=2)
            print('Saved PNG.')
        except Exception as e:
            print('PNG export requires kaleido: pip install -U kaleido')
            print(e)

if __name__ == '__main__':
    main()
