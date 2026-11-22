"""E4-R seam_random (Member C): randomly perturbed boundary control for E4.

Starts from the same E3 rectangle as E4 and reshapes its left/right edges
under exactly the same constraints: offset band [-r, r], at most one pixel
change between rows, min_width (no crossing), image bounds, the same bend
penalty mu, and the same finalize_boundaries fallback/area rule.

The only difference is the cost table: i.i.d. Uniform[0,1) noise instead of
seam costs, solved with the same joint DP. Hence:
  * feasibility is identical to E4 (same constraint set), so too_small and
    infeasible fallbacks match E4 sample by sample;
  * a bend penalty still discourages bends (its own value, see below);
  * the path ignores image content entirely.
The area-budget fallback can differ because the random paths differ; D must
compare area_final, bends and seam_count (boundary length) distributions.

Bend penalty: i.i.d. noise is uncorrelated between rows, while seam costs
are spatially smooth, so with the same mu E4-R bends far more than E4
(about 2x on smooth synthetic images). seam_random_mu therefore gives E4-R
its own penalty, calibrated on shape statistics only (no labels, no model)
with scripts/calibrate_seam_random_mu.py. None means "use seam_mu".

RNG: never advances the global generators, so E3 selection and every later
training draw are unchanged. The noise generator is reseeded from a value
drawn from a clone of the device RNG state XOR a fixed constant, so it is
reproducible under the run seed but not the same stream that B's candidate
translations read from that cloned state.
"""
import torch

from .seam_dp import _solve_and_finalize, edge_constraints, reshape_e3_policy

_STREAM_SALT = 0x5EA3_4A11_D0C0_FFEE  # separates E4-R noise from B's translation stream


def _noise_generator(device):
    clone = torch.Generator(device=device)
    state = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state()
    clone.set_state(state)
    raw = int(torch.randint(0, 2 ** 62, (1,), device=device, generator=clone).item())
    gen = torch.Generator()  # CPU noise, moved to the device afterwards
    gen.manual_seed(raw ^ _STREAM_SALT)
    return gen


def random_boundaries(source, target, boxes, radius=2, mu=0.0, min_width=1,
                      area_budget=0.1, eps=1e-3, generator=None):
    """E4-R core: random connected left/right edges under E4's constraints.

    Returns the same dict as optimize_boundaries (dp_objective is the value
    of the random objective, meaningful only for debugging).
    """
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    n, _, h, w = source.shape
    if generator is None:
        generator = _noise_generator(source.device)
    al, rv = edge_constraints(boxes, h, w, "left", radius)
    ar, _ = edge_constraints(boxes, h, w, "right", radius)
    shape = (n, h, 2 * radius + 1)
    cl = torch.rand(shape, generator=generator).to(source.device)
    cr = torch.rand(shape, generator=generator).to(source.device)
    return _solve_and_finalize(source, target, boxes, cl, al, cr, ar, rv,
                               mu, min_width, area_budget, eps)


def seam_random_cutmix(images, labels, alpha=1.0, p=0.5, k=4, gamma=0.5, beta=0.0,
                       seam_radius=2, seam_mu=0.0, min_width=2, area_budget=0.1,
                       seam_eps=1e-3, generator=None, seam_random_mu=None):
    """E4-R policy: same E3 rectangle and rules as seam_dp, random boundary.

    seam_random_mu: bend penalty for the random paths (None -> seam_mu).
    Metadata records seam_mu (E4's value) and bend_mu (the value used here).
    """
    bend_mu = seam_mu if seam_random_mu is None else seam_random_mu

    def reshape(source, target, boxes):
        return random_boundaries(source, target, boxes, seam_radius, bend_mu,
                                 min_width, area_budget, seam_eps, generator)
    return reshape_e3_policy(images, labels, "seam_random", reshape, alpha, p, k,
                             gamma, beta, seam_radius, seam_mu, min_width,
                             area_budget, seam_eps, bend_mu)
