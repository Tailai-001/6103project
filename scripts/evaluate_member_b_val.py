"""Verify saved B best checkpoints on the fixed clean validation split only."""
import csv
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data.dataset import build_dataloaders
from models.resnet import build_resnet18
from train import normalize_images, set_seed


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suffix",default="")
    parser.add_argument("--num-workers",type=int,default=0)
    args=parser.parse_args()
    if args.num_workers<0: parser.error("num-workers must be nonnegative")
    set_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _,loader=build_dataloaders(num_workers=args.num_workers)
    results=[]
    for policy,k in [('dual_edge',1),('dual_edge',4),('source_edge',4),('dual_edge',8),('source_edge',8)]:
        name=f'b_{policy}_k{k}_s42'+('_'+args.suffix if args.suffix else '')
        state=torch.load(Path('checkpoints')/name/'best.pth',map_location='cpu',weights_only=True)
        model=build_resnet18().to(device)
        model.load_state_dict(state['model_state_dict']);model.eval()
        counts=torch.zeros(100,100,dtype=torch.int64,device=device)
        loss_sum=0.
        for x,y in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            logits=model(normalize_images(x))
            loss_sum+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
            counts+=torch.bincount(100*y+logits.argmax(1),minlength=10000).reshape(100,100)
        counts=counts.cpu();tp=counts.diag().double()
        total=int(counts.sum());accuracy=float(tp.sum())/total
        macro_f1=float((2*tp/(counts.sum(0)+counts.sum(1)).clamp_min(1)).mean())
        with (Path('logs')/(name+'.csv')).open() as f: rows=list(csv.DictReader(f))
        best=max(rows,key=lambda row:float(row['val_accuracy']))
        if total!=5000 or state['epoch']!=int(best['epoch']) or abs(accuracy-float(best['val_accuracy']))>1e-10:
            raise RuntimeError(f'{name}: checkpoint vs logged validation mismatch')
        results.append(dict(run_id=name,split='fixed_validation_only',samples=total,
            num_workers=args.num_workers,evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            checkpoint_epoch=state['epoch'],val_top1_percent=100*accuracy,
            val_macro_f1=macro_f1,val_loss=loss_sum/total,
            logged_accuracy_verified=True))
        del model,state
    Path('results/member_b/checkpoint_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
