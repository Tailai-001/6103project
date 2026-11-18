"""Generate B tables/curves only from completed raw training logs."""
import csv
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--suffix",default="")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=root/'results/member_b'; out.mkdir(parents=True,exist_ok=True)
    summary=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for policy,k in [('dual_edge',1),('dual_edge',4),('source_edge',4),('dual_edge',8),('source_edge',8)]:
        name=f'b_{policy}_k{k}_s42'+('_'+args.suffix if args.suffix else '')
        with (root/f'logs/{name}.csv').open() as f: rows=list(csv.DictReader(f))
        epochs=[int(row['epoch']) for row in rows]
        if epochs!=list(range(1,201)): raise RuntimeError(f'{name}: incomplete or duplicate epochs')
        if any(int(row['train_samples'])!=45000 for row in rows):
            raise RuntimeError(f'{name}: formal run has incomplete training data')
        best=max(rows,key=lambda row:float(row['val_accuracy']))
        seconds=sum(float(row['epoch_seconds']) for row in rows)
        summary.append(dict(run_id=name,policy=policy,k=k,seed=42,epochs=200,
            best_val_top1_percent=100*float(best['val_accuracy']),best_epoch=int(best['epoch']),
            val_loss_at_best=float(best['val_loss']),sum_epoch_seconds=seconds,
            log_path=f'logs/{name}.csv',checkpoint_path=f'checkpoints/{name}/best.pth'))
        ax.plot(epochs,[100*float(row['val_accuracy']) for row in rows],label=f'{policy}, K={k}',linewidth=1)
    ax.set(xlabel='Epoch',ylabel='Clean validation Top-1 (%)',title='CIFAR-100 / CIFAR ResNet-18, seed 42')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'validation_curves.png',dpi=180); plt.close(fig)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 阶段 B 实测结果','',
        '所有数值来自同 seed 42 的完整 200 epoch 原始日志；指标是干净验证集 Top-1，不是 test。单 seed 不报告标准差。',
        '', '| Policy | K | Best val Top-1 (%) | Best epoch | Epoch time sum (min) |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['policy']} | {row['k']} | {row['best_val_top1_percent']:.2f} | {row['best_epoch']} | {row['sum_epoch_seconds']/60:.2f} |")
    k4={row['policy']:row['best_val_top1_percent'] for row in summary if row['k']==4}
    lines += ['', f"K=4 双侧减单侧：{k4['dual_edge']-k4['source_edge']:+.2f} 个百分点。结果允许为负，不预设内容平衡能提高分类准确率。",'',
        'A 已上传基线：E0=77.74%、E1=80.52%（A40 / torch 2.9.1）。跨设备和软件版本比较仅作背景；B 的 K=1 为当前环境的 E1 等价采样控制。',
        '', '配对混合示例见 paired_examples.png，Sobel 和候选见 sobel_candidates.png，收敛曲线见 validation_curves.png。',
        'diagnostics.json 的增强计时为隔离运行、10 次预热、50 次同步重复，包含全部增强操作，不含加载和绘图。',
        '正式任务并发数见 registry.csv 的 notes；并发 walltime 包含 GPU 争用，不能当作隔离训练速度。epoch_seconds 在写入 checkpoint 前记录，求和不包含全部 checkpoint I/O；完整进程 walltime 见登记表。',
        '', '没有进行最终 test 或多 seed 评估；没有据此声称统计显著性。gamma=.5 是预先指定值，未执行额外验证集搜索。']
    (out/'RESULTS_B.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
