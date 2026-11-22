# Boundary quality

Paired, forced-mix audit on CIFAR-100 training images. RGB values are weighted by eligible cross-mask pairs.
Boundary length counts internal four-neighbor pairs. Area uses all inputs. Fallback and bends are over
nonempty base rectangles. Edge snap is over changed masks. N/A indicates no applicable observations.

| Method | RGB all | RGB excluding two padded endpoints | RGB excluding any padded endpoint | Area ratio | Boundary length | Fallback (%) | Bends | Edge snap (%) | N/A after any-padding exclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 | 0.3058 | 0.3150 | 0.2900 | 0.300 | 41.98 | N/A | N/A | N/A | 23 |
| E2 | 0.3066 | 0.3213 | 0.2910 | 0.300 | 54.51 | N/A | N/A | N/A | 14 |
| E3 | 0.2470 | 0.2646 | 0.2543 | 0.300 | 54.72 | N/A | N/A | N/A | 16 |
| E4 | 0.2098 | 0.2242 | 0.2147 | 0.303 | 60.92 | 7.76 | 8.97 | 15.67 | 19 |
| E4-R | 0.2652 | 0.2827 | 0.2614 | 0.294 | 66.81 | 8.43 | 9.20 | 14.49 | 15 |
