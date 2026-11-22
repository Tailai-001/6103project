"""Ensure local MPS support preserves CPU augmentation and the single training loop."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from train import train_one_epoch
from augmentations import cutmix_batch
from models.resnet import build_resnet18


@pytest.mark.parametrize("policy", ["random", "seam_dp", "seam_random"])
def test_explicit_cpu_route_preserves_original_training_step(policy):
    torch.manual_seed(123)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 100))
    reference = copy.deepcopy(model)
    data = [(torch.rand(4, 3, 8, 8), torch.arange(4))]
    cfg = dict(policy=policy, p=1., k=4, gamma=.5, beta=1., seam_radius=2,
               seam_mu=.05, seam_min_width=2, seam_area_budget=.2, seam_random_mu=.225)
    metrics = []
    for item, route in [(reference, None), (model, "cpu")]:
        np.random.seed(43)
        torch.manual_seed(43)
        metrics.append(train_one_epoch(item, data, torch.optim.SGD(item.parameters(), lr=.1),
                                       torch.device("cpu"), cfg, augmentation_device=route))
    assert metrics[0]["loss"] == metrics[1]["loss"]
    assert metrics[0]["samples"] == metrics[1]["samples"] == 4
    for left, right in zip(reference.parameters(), model.parameters()):
        assert torch.equal(left, right)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
@pytest.mark.parametrize("policy", ["random", "seam_dp", "seam_random"])
def test_cpu_augmentation_and_mps_model_can_train(policy):
    torch.manual_seed(43)
    np.random.seed(43)
    device = torch.device("mps")
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 100)).to(device)
    original = [p.detach().clone() for p in model.parameters()]
    cfg = dict(policy=policy, p=1., k=4, gamma=.5, beta=1., seam_radius=2,
               seam_mu=.05, seam_min_width=2, seam_area_budget=.2, seam_random_mu=.225)
    result = train_one_epoch(model, [(torch.rand(4, 3, 8, 8), torch.arange(4))],
                             torch.optim.SGD(model.parameters(), lr=.1), device, cfg,
                             augmentation_device="cpu")
    assert result["samples"] == 4 and np.isfinite(result["loss"])
    assert any(not torch.equal(before, after) for before, after in zip(original, model.parameters()))


@pytest.mark.parametrize("policy", ["dual_edge", "seam_aware", "seam_dp", "seam_random"])
def test_policies_do_not_change_later_base_sampling_or_input_randomness(policy):
    streams=[]
    for arm in ('random',policy):
        torch.manual_seed(43)
        np.random.seed(43)
        batches=[]
        for _ in range(3):
            images=torch.rand(6,3,12,13)
            labels=torch.arange(6)
            meta=cutmix_batch(images,labels,policy=arm,p=.5,k=4,gamma=.5,beta=1,
                               seam_radius=2,seam_mu=.05,seam_min_width=2,
                               seam_area_budget=.2,seam_random_mu=.225)[4]
            batches.append((images,meta['permutation'],meta.get('base_boxes',meta['boxes']),
                            torch.get_rng_state(),np.random.get_state()[1]))
        streams.append(batches)
    for left,right in zip(*streams):
        for a,b in zip(left[:-1],right[:-1]):
            assert torch.equal(a,b)
        assert np.array_equal(left[-1],right[-1])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS unavailable")
@pytest.mark.parametrize("policy", ["random", "seam_dp", "seam_random"])
def test_multistep_resnet_mps_agrees_with_cpu_reference(policy):
    torch.manual_seed(43)
    reference=build_resnet18()
    accelerated=copy.deepcopy(reference).to('mps')
    x=torch.rand(16,3,32,32)
    batches=[(x,torch.arange(16))]*2
    cfg=dict(policy=policy,p=.5,alpha=1.,k=4,gamma=.5,beta=1.,seam_radius=2,
             seam_mu=.05,seam_min_width=2,seam_area_budget=.2,seam_random_mu=.225)
    results=[]
    for dev,model in [('cpu',reference),('mps',accelerated)]:
        torch.manual_seed(43)
        np.random.seed(43)
        results.append(train_one_epoch(model,batches,torch.optim.SGD(model.parameters(),lr=.1,momentum=.9),
                                        torch.device(dev),cfg,augmentation_device='cpu'))
    assert results[0]['loss']==pytest.approx(results[1]['loss'],rel=3e-4,abs=3e-4)
    assert torch.allclose(reference.fc.weight,accelerated.fc.weight.cpu(),atol=1e-3,rtol=1e-3)
