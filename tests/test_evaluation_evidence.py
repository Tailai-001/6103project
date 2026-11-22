import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import verify_checkpoint_log
from scripts.evaluate_member_c_val import evaluate


@pytest.mark.parametrize('invalid', ['nan', 'inf', '-0.1', '1.1'])
def test_checkpoint_selection_rejects_invalid_validation_accuracy(tmp_path, invalid):
    path=tmp_path/'run.csv'
    path.write_text(f'epoch,val_accuracy,train_samples\n1,{invalid},45000\n2,0.8,45000\n')
    state=dict(epoch=1,best_val_accuracy=float(invalid),config=dict(training=dict(epochs=2)))
    with pytest.raises(ValueError,match='finite.*accuracy'):
        verify_checkpoint_log(state,path,require_formal=True)


def test_checkpoint_selection_rejects_nan_checkpoint_metric(tmp_path):
    path=tmp_path/'run.csv'
    path.write_text('epoch,val_accuracy,train_samples\n1,0.8,45000\n')
    with pytest.raises(ValueError,match='finite.*accuracy'):
        verify_checkpoint_log(dict(epoch=1,best_val_accuracy=float('nan')),path)


class FixedLogits(nn.Module):
    def __init__(self,logits):
        super().__init__()
        self.register_buffer('logits',logits)
        self.cursor=0

    def forward(self,x):
        value=self.logits[self.cursor:self.cursor+len(x)]
        self.cursor+=len(x)
        return value


def test_metrics_match_manual_confusion_counts_on_unequal_batches():
    # True [0,0,1,2], predicted [0,1,1,2]: F1=(2/3,2/3,1,0,...).
    logits=torch.zeros(4,100)
    logits[torch.arange(4),torch.tensor([0,1,1,2])]=2
    loader=DataLoader(TensorDataset(torch.rand(4,3,2,2),torch.tensor([0,0,1,2])),batch_size=3)
    n,accuracy,f1,loss=evaluate(FixedLogits(logits).eval(),loader,'cpu')
    assert n==4 and accuracy==.75
    assert f1==pytest.approx((2/3+2/3+1)/100)
    assert loss==pytest.approx(math.log(math.exp(2)+99)-1.5)


def test_empty_evaluation_cannot_be_reported_as_a_score():
    loader=DataLoader(TensorDataset(torch.empty(0,3,2,2),torch.empty(0,dtype=torch.long)))
    with pytest.raises(ValueError,match='empty'):
        evaluate(FixedLogits(torch.empty(0,100)).eval(),loader,'cpu')


def test_nonfinite_predictions_cannot_be_reported_as_a_score():
    loader=DataLoader(TensorDataset(torch.rand(1,3,2,2),torch.zeros(1,dtype=torch.long)))
    with pytest.raises(FloatingPointError,match='Non-finite'):
        evaluate(FixedLogits(torch.full((1,100),float('nan'))).eval(),loader,'cpu')
