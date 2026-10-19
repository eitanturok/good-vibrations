# torch.compile NaN repro (torch 2.11): src7 render()'s amax over peaks gives NaN grads in every upstream weight once a peak is kept,
# even though the mask is not in the loss. Eager is fine; .sum(1), no ray interpolation, or .max(1).values (the fix) are fine.
#   .venv/bin/python scripts/repro_amax_compile_nan.py [B] [full|nointerp|sum|nokeep|max]
import math, sys, torch, torch.nn as nn
torch.manual_seed(0)
B, N, n, H, W = int(sys.argv[1]) if len(sys.argv) > 1 else 2, 4, 36, 64, 64
variant = sys.argv[2] if len(sys.argv) > 2 else "full"

def render(r, idx, keep, tau=0.2):
    cy, cx = (idx // W + 0.5)[..., None, None], (idx % W + 0.5)[..., None, None]
    dy = torch.arange(H, device=idx.device)[:, None] + 0.5 - cy
    dx = torch.arange(W, device=idx.device)[None, :] + 0.5 - cx
    rho, k = torch.hypot(dy, dx), torch.atan2(dy, dx) % (2 * math.pi) / (2 * math.pi / n)
    i0, f = k.floor().long() % n, k - k.floor()
    ray = lambda i: r.gather(2, i.flatten(2)).view_as(rho)
    if variant == "nointerp": inside = torch.sigmoid((ray(i0) - rho) / tau)
    else: inside = torch.sigmoid(((1 - f) * ray(i0) + f * ray((i0 + 1) % n) - rho) / tau)
    if variant == "sum": return (inside * keep[..., None, None]).sum(1)
    if variant == "nokeep": return inside.amax(1)
    if variant == "max": return (inside * keep[..., None, None]).max(1).values
    return (inside * keep[..., None, None]).amax(1)

conv = nn.Conv2d(8, n, 3, padding=1).cuda()
def fwd(x, idx, keep):
    p = conv(x)                                                               # (B,n,H,W) log ray lengths
    r = p.flatten(2).gather(2, idx[:, None].expand(-1, n, -1)).transpose(1, 2).clamp(-3, 10).exp()  # (B,N,n) rays at the peaks
    return p, render(r, idx, keep)
x = torch.randn(B, 8, H, W, device="cuda")
idx = torch.randint(0, H * W, (B, N), device="cuda")
keep = torch.zeros(B, N, device="cuda"); keep[0, 0] = 1                       # one kept peak, as in the failing step
for name, f in (("eager   ", fwd), ("compiled", torch.compile(fwd))):
    conv.zero_grad()
    p, mask = f(x, idx, keep)
    p.square().mean().backward()                                              # the loss uses p, not the mask
    print(f"B={B} {variant:8s} {name}: mask range ({mask.min().item():.2f},{mask.max().item():.2f}) non-finite conv grads: {int((~conv.weight.grad.isfinite()).sum())}")
