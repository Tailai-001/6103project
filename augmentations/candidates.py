"""Area-preserving candidates using A's unchanged E1 sampler."""
import torch
from .cutmix import random_cutmix


def translate_candidates(base_boxes, height, width, k, generator):
    """Candidate 0 is original; others are uniform valid translations.

    Duplicates allowed, first-index tie breaking. Draw (K-1)×N×2 so
    per-image prefixes match when K changes. Caller owns RNG generator.
    """
    n = base_boxes.shape[0]
    size = base_boxes[:,2:] - base_boxes[:,:2]
    if k == 1:
        return base_boxes[:, None]
    # Fixed draw shape for each candidate also preserves prefixes on CUDA,
    # where the RNG kernel's mapping can depend on the requested tensor size.
    u = torch.stack([torch.rand((n,2), device=base_boxes.device, generator=generator)
                     for _ in range(k-1)], dim=1)
    extent = base_boxes.new_tensor([height,width]) - size + 1
    starts = (u * extent[:,None]).floor().long()
    extra = torch.cat((starts, starts + size[:,None]), -1)
    return torch.cat((base_boxes[:,None], extra), 1)


def sample_candidates(images, labels, alpha=1.0, p=0.5, k=4):
    """Preserve A's torch+numpy draws exactly; extra draws use cloned RNG.

    Does not advance global RNG for translations, so K/policy cannot alter
    subsequent base boxes, training transformations or source permutations.
    Returns candidates and A metadata; all policies share this API.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    _, _, _, _, metadata = random_cutmix(images, labels, alpha, p)
    device = images.device
    generator = torch.Generator(device=device)
    state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    generator.set_state(state)
    candidates = translate_candidates(metadata["boxes"], *images.shape[-2:], k, generator)
    return candidates, metadata


def boxes_to_mask(boxes, height, width):
    y = torch.arange(height, device=boxes.device)[None,:,None]
    x = torch.arange(width, device=boxes.device)[None,None,:]
    t,l,b,r = boxes.unbind(-1)
    return ((y >= t[:,None,None]) & (y < b[:,None,None]) &
            (x >= l[:,None,None]) & (x < r[:,None,None]))[:,None]
