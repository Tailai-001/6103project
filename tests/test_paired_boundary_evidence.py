import numpy as np
import pytest

from analysis.paired_boundary_evidence import bootstrap_difference, eligible_pair


def test_paired_identical_observations_have_zero_interval():
    values=np.array([1.,2.,4.,7.])
    delta,ci=bootstrap_difference(values,np.ones(4),values,np.ones(4),[0,0,1,1],500)
    assert delta==0 and ci==[0,0]


def test_batch_resampling_preserves_pairing_and_known_ratio_difference():
    counts=np.array([2.,3.,4.,5.])
    delta,ci=bootstrap_difference(.2*counts,counts,.5*counts,counts,[0,0,1,1],500)
    assert delta==pytest.approx(-.3)
    assert ci==pytest.approx([-.3,-.3])


def test_single_cluster_does_not_claim_an_independent_interval():
    delta,ci=bootstrap_difference([1,2],[2,4],[2,4],[2,4],[0,0])
    assert delta==-.5 and ci is None


def test_shape_filter_does_not_look_at_rgb_outcome():
    a=dict(status='accepted',area_ratio=.3,count_all=50,bends=9,count_all_valid=50,
           touched_edge=0,sum_all=999)
    b=dict(a,area_ratio=.31,count_all=54,bends=10,sum_all=0)
    assert eligible_pair(a,b,'all','accepted_shape_close')
    assert eligible_pair(dict(a,sum_all=0),dict(b,sum_all=999),'all','accepted_shape_close')
    assert not eligible_pair(a,dict(b,count_all=70),'all','accepted_shape_close')
    assert not eligible_pair(a,dict(b,touched_edge=1),'all','accepted_shape_close')
    assert not eligible_pair(a,dict(b,status='area'),'all','accepted_shape_close')
