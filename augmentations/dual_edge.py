"""dual_edge public policy."""
from .selection import edge_cutmix


def dual_edge_cutmix(images, labels, alpha=1.0, p=0.5, k=4, gamma=0.5):
    return edge_cutmix(images, labels, "dual_edge", alpha, p, k, gamma)
