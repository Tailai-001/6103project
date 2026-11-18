"""CIFAR train-only paired examples, complete augmentation timing, mask statistics."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor
from augmentations import cutmix_batch
from augmentations.edge_utils import sobel_edges, candidate_edge_scores


def reset():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",default="data/cifar100")
    p.add_argument("--output",default="results/member_b")
    args=p.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    dataset=CIFAR100(args.data_root,train=True,download=False,transform=ToTensor())
    # Fixed A training indices only; never use validation/test images for illustrations.
    ids=np.load("data/splits/train_indices.npy")[:128]
    device="cuda" if torch.cuda.is_available() else "cpu"
    x=torch.stack([dataset[int(i)][0] for i in ids]).to(device)
    y=torch.tensor([dataset[int(i)][1] for i in ids],device=device)
    methods=[("random",1),("dual_edge",1),("source_edge",4),("dual_edge",4),("source_edge",8),("dual_edge",8)]
    reports=[]; outputs={}
    for policy,k in methods:
        reset()
        result=cutmix_batch(x,y,policy=policy,k=k,p=1)
        outputs[(policy,k)]=result
        mask=result[4]["mask"]
        perimeter=((mask[:,:,1:]!=mask[:,:,:-1]).flatten(1).sum(1)+
                   (mask[:,:,:,1:]!=mask[:,:,:,:-1]).flatten(1).sum(1))
        # Timing includes normal p=.5 sampling, Sobel, scores, masking and copying.
        for _ in range(10): cutmix_batch(x,y,policy=policy,k=k,p=.5)
        sync(); start=time.perf_counter()
        for _ in range(50): cutmix_batch(x,y,policy=policy,k=k,p=.5)
        sync(); ms=(time.perf_counter()-start)*1000/50
        src_edge,tgt_edge,_=candidate_edge_scores(x,result[4]["permutation"],result[4]["boxes"][:,None])
        reports.append(dict(policy=policy,k=k,augmentation_ms_per_batch=ms,
            selected_source_edge_mean=float(src_edge.mean()),
            selected_target_edge_mean=float(tgt_edge.mean()),
            diagnostic_p=1,timing_p=.5,area_mean=float(mask.float().mean()),
            area_std=float(mask.flatten(1).float().mean(1).std()),
            internal_boundary_length_mean=float(perimeter.float().mean())))
    # Show same target/source pair and base geometry for every method.
    edges=sobel_edges(x).cpu()
    fig,axes=plt.subplots(4,7,figsize=(17,10))
    for row,i in enumerate([0,1,2,3]):
        ref=outputs[("dual_edge",4)][4]; src=int(ref["permutation"][i])
        images=[x[i],x[src]]+[outputs[key][0][i] for key in methods if key!=("dual_edge",1)]
        titles=["Target", "Source", "E1 random", "E2-S K=4", "E2 K=4", "E2-S K=8", "E2 K=8"]
        boxes=[None,None]+[outputs[key][4]["boxes"][i].cpu().tolist() for key in methods if key!=("dual_edge",1)]
        for ax,img,title,box in zip(axes[row],images,titles,boxes):
            ax.imshow(img.detach().cpu().permute(1,2,0)); ax.set_title(title); ax.axis("off")
            if box is not None:
                t,l,b,r=box; ax.add_patch(Rectangle((l-.5,t-.5),r-l,b-t,fill=False,edgecolor="red",linewidth=1.3))
    fig.tight_layout(); fig.savefig(out/"paired_examples.png",dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for row,i in enumerate([0,1]):
        meta=outputs[("dual_edge",4)][4]; src=int(meta["permutation"][i])
        for ax,img,title in zip(axes[row],[x[i].cpu().permute(1,2,0),edges[i],edges[src]],
                               ["Target RGB","Normalized target Sobel","Normalized source Sobel"]):
            ax.imshow(img,cmap="magma" if img.ndim==2 else None); ax.set_title(title); ax.axis("off")
        for j,(t,l,b,r) in enumerate(meta["candidates"][i].cpu().tolist()):
            axes[row,1].add_patch(Rectangle((l-.5,t-.5),r-l,b-t,fill=False,edgecolor=["cyan","lime","white","red"][j]))
    fig.tight_layout(); fig.savefig(out/"sobel_candidates.png",dpi=170); plt.close(fig)
    payload=dict(device=device,gpu=torch.cuda.get_device_name(0) if device=="cuda" else None,
        torch=torch.__version__,dtype=str(x.dtype),batch_size=len(x),warmup=10,repetitions=50,
        timing="synchronized, complete augmentation only, excludes data loading and plotting",results=reports)
    (out/"diagnostics.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
