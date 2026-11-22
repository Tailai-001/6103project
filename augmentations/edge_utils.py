"""B/C shared Sobel and integral-image rectangle scores, RGB before normalization."""
import torch
import torch.nn.functional as F


def sobel_edges(images, eps=1e-6):
    """Return N×H×W magnitudes divided by each image's mean magnitude.

    BT.601 grayscale, replicate padding, unscaled 3×3 Sobel, L2 magnitude.
    Zero/constant images remain zero. Half/bfloat inputs accumulate in float32.
    No candidate-specific normalization; eps is in raw Sobel units.
    """
    if images.ndim != 4 or images.shape[1] != 3 or not images.is_floating_point():
        raise ValueError("Expected floating RGB [N,3,H,W]")
    if eps <= 0:
        raise ValueError("eps must be positive")
    x = images if images.dtype == torch.float64 else images.float()
    gray = (x * x.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
    # Difference form gives exactly zero for constant images, including fp rounding.
    padded = F.pad(gray, (1,1,1,1), mode="replicate")
    gx = ((padded[:,:,0:-2,2:] - padded[:,:,0:-2,:-2]) +
          2*(padded[:,:,1:-1,2:] - padded[:,:,1:-1,:-2]) +
          (padded[:,:,2:,2:] - padded[:,:,2:,:-2]))
    gy = ((padded[:,:,2:,0:-2] - padded[:,:,:-2,0:-2]) +
          2*(padded[:,:,2:,1:-1] - padded[:,:,:-2,1:-1]) +
          (padded[:,:,2:,2:] - padded[:,:,:-2,2:]))
    magnitude = torch.linalg.vector_norm(torch.cat((gx, gy), 1), dim=1)
    return magnitude / magnitude.mean((-2,-1), keepdim=True).clamp_min(eps)


def rectangle_means(edge_maps, candidates):
    """N×K scores from N×H×W maps and int64 N×K×4 half-open boxes.

    Float64 integral accumulation avoids cancellation for tiny low-edge crops.
    Scores retain edge_maps dtype. Cost O(NHW+NK); empty rectangles score zero.
    """
    integral = F.pad(edge_maps.double().cumsum(-2).cumsum(-1), (1,0,1,0))
    top, left, bottom, right = candidates.unbind(-1)
    row = torch.arange(edge_maps.shape[0], device=edge_maps.device)[:, None]
    sums = (integral[row,bottom,right] - integral[row,top,right]
            - integral[row,bottom,left] + integral[row,top,left])
    area = (bottom-top)*(right-left)
    return torch.where(area > 0, sums / area.clamp_min(1), torch.zeros_like(sums)).to(edge_maps.dtype)


def candidate_edge_scores(images, permutation, candidates, gamma=0.5):
    """Stable C interface: source means, target means, source-gamma*target."""
    edges = sobel_edges(images)
    target = rectangle_means(edges, candidates)
    source = rectangle_means(edges[permutation], candidates)
    return source, target, source - gamma * target
