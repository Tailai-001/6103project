"""Run: python -m unittest discover -s tests -v (CPU, plus CUDA if present)."""
import unittest
import numpy as np
import torch
from augmentations import cutmix_batch
from augmentations.cutmix import random_cutmix
from augmentations.edge_utils import sobel_edges, rectangle_means
from augmentations.candidates import boxes_to_mask, translate_candidates


def seed():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


class MemberBTests(unittest.TestCase):
    def setUp(self):
        seed()
        self.x = torch.rand(12,3,17,23)
        self.y = torch.arange(12)

    def call(self, policy="dual_edge", **kw):
        seed()
        return cutmix_batch(self.x, self.y, policy=policy, **kw)

    def test_sobel_constant_and_known_ramp(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            edge = sobel_edges(torch.full((2,3,5,7), 0.7, dtype=dtype))
            self.assertTrue(torch.equal(edge, torch.zeros_like(edge)))
        ramp = torch.arange(7).float()[None,None,None,:].expand(1,3,5,7)
        edge = sobel_edges(ramp)
        self.assertTrue(torch.allclose(edge[0,2], torch.tensor([.5,1,1,1,1,1,.5])*7/6))

    def test_integral_against_brute_force(self):
        maps = torch.rand(2,5,7, dtype=torch.float64)
        boxes = torch.tensor([[[0,0,5,7],[1,2,4,6],[0,0,0,0]],[[0,0,5,7],[4,6,5,7],[2,2,2,4]]])
        scores = rectangle_means(maps, boxes)
        for i in range(2):
            for j, (t,l,b,r) in enumerate(boxes[i].tolist()):
                expected = maps[i,t:b,l:r].mean() if b>t and r>l else torch.tensor(0., dtype=maps.dtype)
                self.assertTrue(torch.allclose(scores[i,j], expected, atol=1e-12))

    def test_integral_tiny_edge_after_large_prefix(self):
        maps=torch.ones(1,32,32)
        maps[0,-1,-1]=1e-6
        boxes=torch.tensor([[[31,31,32,32]]])
        self.assertTrue(torch.allclose(rectangle_means(maps,boxes),torch.tensor([[1e-6]]),atol=1e-10))

    def test_k1_exact_a_e1(self):
        seed()
        old = random_cutmix(self.x, self.y, p=0.5)
        new = self.call(k=1)
        for i in range(4):
            self.assertTrue(torch.equal(old[i], new[i]))
        self.assertTrue(torch.equal(old[4]["boxes"], new[4]["boxes"]))

    def test_gamma_zero_source_equivalence(self):
        a,b = self.call("source_edge", p=1), self.call(gamma=0, p=1)
        for i in range(4):
            self.assertTrue(torch.equal(a[i], b[i]))
        self.assertTrue(torch.equal(a[4]["candidate_scores"], b[4]["candidate_scores"]))

    def test_pixels_labels_area_and_no_mutation(self):
        before = self.x.clone()
        mixed, ya, yb, lam, m = self.call(p=1, k=8)
        self.assertTrue(torch.equal(self.x, before))
        self.assertTrue(torch.equal(mixed, torch.where(m["mask"], before[m["permutation"]], before)))
        self.assertTrue(torch.equal(yb, self.y[m["permutation"]]))
        self.assertTrue(torch.equal(ya,self.y))
        self.assertTrue(torch.equal(lam, 1-m["mask"].flatten(1).float().mean(1)))
        self.assertEqual(m["mask"].dtype, torch.bool)

    def test_candidates_equal_area_and_within_bounds(self):
        m = self.call(p=1,k=8)[4]
        t,l,b,r = m["candidates"].unbind(-1)
        self.assertTrue(((t>=0)&(l>=0)&(b<=17)&(r<=23)&(b>=t)&(r>=l)).all())
        areas=(b-t)*(r-l)
        self.assertTrue(torch.equal(areas, areas[:,:1].expand_as(areas)))
        self.assertTrue(torch.equal(m["base_boxes"],m["candidates"][:,0]))

    def test_prefix_and_rng_independent_of_k(self):
        a=self.call(p=1,k=4)[4]
        state4=torch.get_rng_state()
        np4=np.random.get_state()
        b=self.call(p=1,k=8)[4]
        self.assertTrue(torch.equal(a["candidates"],b["candidates"][:,:4]))
        self.assertTrue(torch.equal(state4,torch.get_rng_state()))
        self.assertTrue(np.array_equal(np4[1], np.random.get_state()[1]))

    def test_argmax_matches_scores(self):
        m = self.call(p=1)[4]
        self.assertTrue(torch.equal(m["selected_indices"],m["candidate_scores"].argmax(1)))
        self.assertTrue(torch.allclose(m["candidate_scores"],m["source_edge_scores"]-.5*m["target_edge_scores"]))

    def test_tie_first_and_no_nan(self):
        self.x.zero_()
        m=self.call(p=1)[4]
        self.assertTrue((m["selected_indices"]==0).all())
        self.assertTrue(torch.isfinite(m["candidate_scores"]).all())

    def test_p0_and_none(self):
        for policy in ["none","random","source_edge","dual_edge"]:
            mixed,_,_,lam,m=self.call(policy,p=0)
            self.assertTrue(torch.equal(mixed,self.x))
            self.assertTrue((lam==1).all())
            self.assertFalse(m["mask"].any())

    def test_full_empty_boundary_translation(self):
        boxes=torch.tensor([[0,0,5,7],[0,0,0,0],[4,6,5,7],[0,0,1,1]])
        gen=torch.Generator().manual_seed(1)
        c=translate_candidates(boxes,5,7,8,gen)
        self.assertTrue(torch.equal(c[0],boxes[0].expand(8,4)))
        mask=boxes_to_mask(boxes,5,7)
        self.assertEqual(mask.flatten(1).sum(1).tolist(),[35,0,1,1])

    def test_singleton_tiny_and_dtype(self):
        for shape in [(1,3,1,1),(1,3,2,3)]:
            for dtype in [torch.float16,torch.bfloat16,torch.float32,torch.float64]:
                self.x=torch.rand(shape).to(dtype)
                self.y=torch.tensor([0])
                mixed,_,_,lam,m=self.call(p=1)
                self.assertEqual(mixed.dtype,dtype)
                self.assertEqual(lam.dtype,dtype)
                self.assertTrue(torch.equal(mixed,self.x))
                self.assertTrue(torch.isfinite(m["candidate_scores"]).all())

    def test_known_source_and_target_preference(self):
        from unittest.mock import patch
        from augmentations.selection import edge_cutmix
        x=torch.zeros(2,3,12,12)
        # Source has a stronger top edge and a weaker bottom edge.
        x[1,:,0:5,6:]=1
        x[1,:,7:12,6:]=.4
        # Target contains only the top edge, so a large penalty selects bottom.
        x[0,:,0:5,6:]=1
        boxes=torch.tensor([[[1,4,4,8],[8,4,11,8]]]*2)
        base={"permutation":torch.tensor([1,0]),"boxes":boxes[:,0]}
        with patch("augmentations.selection.sample_candidates",return_value=(boxes,base)):
            source=edge_cutmix(x,torch.arange(2),"source_edge",k=2)[4]
            dual=edge_cutmix(x,torch.arange(2),"dual_edge",k=2,gamma=10)[4]
        self.assertEqual(source["selected_indices"][0].item(),0)
        self.assertEqual(dual["selected_indices"][0].item(),1)

    def test_invalid_parameters(self):
        for kw in [{"k":0},{"k":1.5},{"k":True},{"alpha":0},{"p":2},{"gamma":float("nan")},{"gamma":-1}]:
            with self.assertRaises(ValueError): self.call(**kw)
        with self.assertRaises(NotImplementedError): self.call("seam_aware")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_dtype_and_degeneracy(self):
        self.x=self.x.cuda().half()
        self.y=self.y.cuda()
        a=self.call("source_edge",p=1)
        b=self.call(gamma=0,p=1)
        for i in range(4): self.assertTrue(torch.equal(a[i],b[i]))
        self.assertEqual(a[0].device,self.x.device)
        self.assertEqual(a[3].dtype,self.x.dtype)
        self.assertTrue(torch.isfinite(a[4]["candidate_scores"]).all())
        self.x=torch.rand(128,3,17,23,device="cuda")
        self.y=torch.arange(128,device="cuda")
        c4=self.call(p=1,k=4)[4]["candidates"]
        c8=self.call(p=1,k=8)[4]["candidates"]
        self.assertTrue(torch.equal(c4,c8[:,:4]))
        seed()
        old=random_cutmix(self.x,self.y)
        new=self.call(k=1)
        for i in range(4): self.assertTrue(torch.equal(old[i],new[i]))


if __name__ == "__main__": unittest.main()
