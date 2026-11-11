# nb136 Q1, 4th method: the flatten readouts (MergeBlock) that merge tokens into a vector:
#   encoder.freq_enc.read.lin.weight (256, 18*256): band slot b -> per-laser token; encoder.laser_enc.read.lin.weight (512, 100*256): laser slot l -> latent u.
# weight variant: ||W[:, slot] * gamma_LN||_F (LN folded, box-agnostic); contribution variant: mean over samples (and lasers) of ||W_slot LN(h_slot)||_2 per box.
# top-K keep curves (same halves / K grid as q1.npz) -> runs/_src9_cache/nb136/q1_merge.npz
import sys, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset
from arch import transformer4 as T4
OUT = ROOT / "runs/_src9_cache/nb136"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""   # "_single" -> read q1_single.npz, write *_single.npz
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
_, ev, _, info = build_dataset("tukey", num_workers=0)
model = T4.Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
model.load_state_dict(torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
DS = ev[0].dataloader.dataloader.dataset.dataset
B, Q = np.load(OUT / "base.npz"), np.load(OUT / f"q1{SUF}.npz")
fr, lr = model.encoder.freq_enc.read, model.encoder.laser_enc.read
def wnorm(m, n): W = (m.lin.weight.view(m.lin.weight.shape[0], n, -1) * m.norm.weight).detach(); return W.norm(dim=(0, 2)).cpu().numpy()
w_b, w_l = wnorm(fr, 18), wnorm(lr, 100)

REC = {}
def rec_forward(self, x, m, name):  # MergeBlock.forward + per-slot contribution norms ||W_i LN(x_i)||
    h = self.norm(x).masked_fill(~m[..., None], 0)                               # (..., n, D)
    W = self.lin.weight.view(self.lin.weight.shape[0], h.shape[-2], -1)          # (out, n, D)
    REC[name] = torch.einsum("ond,...nd->...no", W, h).norm(dim=-1)              # (..., n)
    s = torch.nn.functional.linear(h.flatten(-2), self.lin.weight)
    if self.rescale: s = s * (m.shape[-1] / m.sum(-1, keepdim=True)).to(s.dtype)
    return s + self.lin.bias
fr.forward = lambda x, m: rec_forward(fr, x, m, "band"); lr.forward = lambda x, m: rec_forward(lr, x, m, "laser")

sel, box, half = Q["sel"], Q["box"], Q["half"]
X = torch.cat([torch.stack([DS[int(B["i"][j])]["fft"] for j in sel[s:s + 200]]).cuda() for s in range(0, len(sel), 200)]); Y = torch.from_numpy(B["y"][sel]).cuda()  # X chunked to GPU: CPU peak stays small
cb, cl = [], []
with torch.no_grad():
    for s in range(0, len(X), 50):
        x = X[s:s + 50].cuda(); n = len(x)
        model.encoder(x, x.new_ones(n, 100, dtype=torch.bool), x.new_ones(n, 18, dtype=torch.bool))
        cb.append(REC["band"].mean(1).cpu()); cl.append(REC["laser"].cpu())        # band: (n,100,18) -> mean over lasers
c_b, c_l = torch.cat(cb).numpy(), torch.cat(cl).numpy()

def fwd(x, m_l, m_b): return model.decoder(model.encoder(x, m_l, m_b)).sigmoid()
curves = {}
with torch.no_grad():
    for b in ("plastic", "gastro", "cardboard"):
        a, e = (box == b) & (half == 0), np.nonzero((box == b) & (half == 1))[0]
        xe, ye, n = X[e].cuda(), Y[e], len(e)
        orders = {"weight": (np.argsort(-w_b), np.argsort(-w_l)), "contribution": (np.argsort(-c_b[a].mean(0)), np.argsort(-c_l[a].mean(0)))}
        for nm, (ob, ol) in orders.items():
            for unit, o, Ks, T in (("band", ob, Q["KB"], 18), ("laser", ol, Q["KL"], 100)):
                r = []
                for k in Ks:
                    m = torch.zeros(n, T, dtype=torch.bool, device="cuda"); m[:, torch.as_tensor(o[:k].copy())] = True
                    one = torch.ones(n, 118 - T, dtype=torch.bool, device="cuda")
                    ml, mb = (one, m) if unit == "band" else (m, one)
                    p = torch.cat([fwd(xe[s:s + 100], ml[s:s + 100], mb[s:s + 100]) for s in range(0, n, 100)]) > .08
                    r.append(((p & ye).flatten(1).sum(1) / (p | ye).flatten(1).sum(1).clamp_min(1)).mean().item())
                curves[f"c|{b}|{unit}|{nm}"] = np.array(r)
        print(b, {k: v[[0, len(v) // 2, -1]].round(3).tolist() for k, v in curves.items() if f"|{b}|" in k}, flush=True)
np.savez(OUT / f"q1_merge{SUF}.npz", w_b=w_b, w_l=w_l, c_b=c_b, c_l=c_l, **curves)
print("done")
