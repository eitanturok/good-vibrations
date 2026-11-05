"""trace a src9 arch's forward on a dummy batch -> one self-contained HTML dataflow diagram (modules, functional ops, tensor shapes, params).
  .venv/bin/python scripts/arch_diagram.py                                         # src9/arch/transformer4.py -> arch_transformer4.html
  .venv/bin/python scripts/arch_diagram.py --arch src9/arch/hybrid.py --out /tmp/hybrid.html
  .venv/bin/python scripts/arch_diagram.py --watch                                 # re-render whenever the arch file (or run.py) changes
  .venv/bin/python scripts/arch_diagram.py --arch src9/arch/transformer3.py -- --d 128 --heads 4   # model knobs, as run.py flags
generic: the model class = the class in the arch file whose __init__ takes data_info first; its kwargs = run.py's DEFAULTS[<file stem>] that the
ctor accepts (+ the overrides after --). data_info = shapes / lists from the newest wandb config + the newest MDS cache's stats.pt (synthetic if absent).
capture: forward pre/post hooks on every module + a TorchFunctionMode for the functional ops between them; tensors are followed by identity, so the
diagram is the real dataflow (branches, residuals), not just call order. CPU, eval mode, no_grad, eager. On any failure: an HTML page with the traceback."""
import argparse, ast, html, importlib.util, inspect, json, math, os, re, subprocess, sys, time, traceback, weakref
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src9"

p = argparse.ArgumentParser()
p.add_argument("--arch", default="src9/arch/transformer4.py")
p.add_argument("--out", default=None, help="default: arch_<stem>.html in the cwd")
p.add_argument("--model", default=None, help="run.py DEFAULTS key (default: the arch file's stem)")
p.add_argument("--batch", type=int, default=6, help="dummy batch size (6: no clash with the other dims, so B is named unambiguously)")
p.add_argument("--watch", action="store_true")
argv = sys.argv[1:]
knobs = argv[argv.index("--") + 1:] if "--" in argv else []
args = p.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
arch = (ROOT / args.arch).resolve() if not Path(args.arch).is_absolute() else Path(args.arch)
out_path = Path(args.out or f"arch_{arch.stem}.html").resolve()

#*** 1 watch: re-run this script (fresh process) whenever the arch file or run.py changes

if args.watch:
    cmd = [sys.executable, __file__] + [a for a in sys.argv[1:] if a != "--watch"]
    last = None
    print(f"watching {arch} -> {out_path} (ctrl-c to stop)")
    while True:
        m = tuple(f.stat().st_mtime if f.exists() else 0 for f in (arch, SRC / "run.py"))
        if m != last:
            time.sleep(1.5)                                                            # let the writer finish
            if tuple(f.stat().st_mtime if f.exists() else 0 for f in (arch, SRC / "run.py")) != m: continue
            print(f"[{datetime.now():%H:%M:%S}] rendering ..."); subprocess.run(cmd); last = m
        time.sleep(2)

#*** 2 build the model the way run.py does

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, str(SRC))
import torch, torch.nn as nn
from torch.overrides import TorchFunctionMode

def run_defaults():
    """run.py's COMMON / AUGS / DEFAULTS, by executing only those top-level assignments (importing run.py would import every arch)"""
    ns, keep = {}, ("COMMON", "AUGS", "DEFAULTS")
    for st in ast.parse((SRC / "run.py").read_text()).body:
        if isinstance(st, ast.Assign) and any((isinstance(t, ast.Name) and t.id in keep) or (isinstance(t, ast.Subscript) and getattr(t.value, "id", "") in keep) for t in st.targets):
            exec(compile(ast.Module([st], []), "run.py", "exec"), ns)
    return ns.get("DEFAULTS", {}), ns.get("AUGS", {}), ns.get("COMMON", {})

def data_info():
    """the data_info dict run.py gets from build_dataset, without building the dataset: lists from the newest wandb config, stats from the newest MDS cache"""
    info = dict(n_laser_rows=10, n_laser_cols=10, n_freqs=1170, n_channels=2, freq_range=(100.0, 1000.0), n_objects=[0, 1, 2],
                objects=["", "candle", "cylinder-1000g", "cylinder-100g", "cylinder-200g", "cylinder-500g", "glass-cup", "green-cube+red-cube", "red-cube", "red-cube+vase", "soap dispenser", "vase", "unseen"],
                boxes=["cardboard", "gastronorm", "green-plastic"])
    src = ["defaults"]
    try:
        import yaml
        for f in sorted(ROOT.glob("wandb/run-*/files/config.yaml"), key=lambda f: f.stat().st_mtime, reverse=True)[:30]:
            c = yaml.safe_load(f.read_text())
            if "objects" in c and "boxes" in c:
                for k in ("n_laser_rows", "n_laser_cols", "n_freqs", "n_channels", "freq_range", "n_objects", "objects", "boxes"):
                    if k in c: info[k] = c[k]["value"]
                info["freq_range"] = tuple(float(v) for v in info["freq_range"]); src = [f"wandb {f.parts[-3]}"]; break
    except Exception: pass
    caches = sorted((ROOT / "experiments/mds_src9").glob("cache_*/stats.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
    if caches: info |= torch.load(caches[0], mmap=True, weights_only=False); src.append(f"stats {caches[0].parent.name}")
    else:                                                                              # synthetic stats with the right shapes
        nb, L, F, C = len(info["boxes"]), info["n_laser_rows"] * info["n_laser_cols"], info["n_freqs"], info["n_channels"]
        z, o = torch.zeros(nb, L, F, C), torch.ones(nb, L, F, C)
        info |= dict(cum_mean=z, cum_var=o, empty_mean=z, empty_std=torch.ones(nb, F, C), empty5_mean=z, empty5_std=o, n_empty=[1.0] * nb, empty5_idx={})
        src.append("synthetic stats")
    return info, src

def find_model_class(mod):
    """the nn.Module defined in the arch file whose __init__'s first argument is the data info"""
    cands = [c for c in vars(mod).values() if inspect.isclass(c) and issubclass(c, nn.Module) and c.__module__ == mod.__name__]
    first = lambda c: next(iter(list(inspect.signature(c.__init__).parameters)[1:]), "")
    good = [c for c in cands if "info" in first(c)] or [c for c in cands if "ComposerModel" in [b.__name__ for b in c.__mro__]]
    assert good, f"no model class in {arch.name}: want an nn.Module whose __init__ takes data_info first (classes: {[c.__name__ for c in cands]})"
    return sorted(good, key=lambda c: inspect.getsourcelines(c)[1])[-1]

def parse_knobs(knobs, types):
    """['--d', '128', '--rope', '1'] -> {'d': 128, 'rope': 1}, typed like the defaults"""
    out = {}
    for k, v in zip(knobs[::2], knobs[1::2]):
        k = k.lstrip("-").replace("-", "_"); t = types.get(k)
        out[k] = v if t is None or isinstance(t, str) else type(t)(float(v)) if isinstance(t, (int, bool)) else type(t)(v)
    return out

def dummy_batch(info, B, H, W):
    g = torch.Generator().manual_seed(0)
    nb, K = len(info["boxes"]), max(info["n_objects"])
    box = torch.arange(B) % nb
    mean, std = info["cum_mean"][box].float(), info["cum_var"][box].float().sqrt()
    fft = (mean + std * torch.randn(mean.shape, generator=g)).abs()                   # (B,L,F,C), a plausible spectrum per box
    mask = (torch.rand(B, H, W, generator=g) > 0.8).float()
    return dict(fft=fft, box=box, mask_true=mask, inst=torch.zeros(B, K, H, W), objs=torch.zeros(B, K, H, W), sdf=torch.zeros(B, H, W),
                object=torch.arange(B) % (len(info["objects"]) - 1), coms=torch.rand(B, K, 2, generator=g), n_objects=torch.arange(B) % (K + 1),
                experiment=["x"] * B, position_id=torch.arange(B), speaker=torch.ones(B, dtype=torch.long), sample_id=torch.arange(B), layout=["x"] * B,
                crop_h=torch.full((B,), H), crop_w=torch.full((B,), W))

#*** 3 trace

TRANSPARENT = {"view", "reshape", "permute", "transpose", "flatten", "unflatten", "contiguous", "__getitem__", "squeeze", "unsqueeze", "to", "float", "half", "bfloat16",
               "double", "type", "type_as", "expand", "expand_as", "chunk", "split", "unbind", "__iter__", "detach", "clone", "t", "movedim", "moveaxis", "view_as", "narrow",
               "select", "repeat", "bool", "long", "int", "byte", "repeat_interleave", "split_with_sizes", "tensor_split", "swapaxes", "mT", "T", "real", "imag", "requires_grad_", "cpu", "cuda", "broadcast_to"}
CONSTANT = {"new_ones", "new_zeros", "new_full", "new_empty", "new_tensor", "zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like", "randint_like"}
SKIP = {"__get__", "__set__", "size", "dim", "__len__", "numel", "__format__", "__repr__", "__bool__", "item", "tolist", "__index__", "__int__", "__float__", "__hash__",
        "__setitem__", "__contains__", "__eq__", "__ne__", "__lt__", "__gt__", "__le__", "__ge__", "any", "all", "is_floating_point", "is_complex", "element_size", "data_ptr",
        "stride", "storage_offset", "is_contiguous", "__reduce_ex__", "__deepcopy__", "__array__", "numpy", "ndimension", "nelement", "get_device", "__torch_function__"}
BINARY = {"add", "sub", "mul", "div", "truediv", "true_divide", "floordiv", "pow", "radd", "rsub", "rmul", "rdiv", "rtruediv", "rpow", "iadd", "isub", "imul", "idiv", "itruediv", "add_", "sub_", "mul_", "div_"}
SYMBOL = {"add": "+", "radd": "+", "iadd": "+", "add_": "+", "sub": "−", "rsub": "−", "isub": "−", "sub_": "−", "mul": "×", "rmul": "×", "imul": "×", "mul_": "×",
          "div": "÷", "truediv": "÷", "true_divide": "÷", "rdiv": "÷", "rtruediv": "÷", "itruediv": "÷", "div_": "÷", "matmul": "@", "rmatmul": "@", "pow": "^"}
OPNAME = {"scaled_dot_product_attention": "attention", "cat": "concat", "concat": "concat", "softmax": "softmax", "log_softmax": "log-softmax", "einsum": "einsum",
          "layer_norm": "layer-norm", "rms_norm": "rms-norm", "normalize": "l2-normalize", "rsqrt": "rsqrt", "sum": "Σ sum", "sigmoid": "σ sigmoid", "linear": "linear"}

def flat(x):
    if isinstance(x, torch.Tensor): return [x]
    if isinstance(x, dict): return [t for v in x.values() for t in flat(v)]
    if isinstance(x, (list, tuple)): return [t for v in x for t in flat(v)]
    return []

def fname(func):
    n = getattr(func, "__name__", None) or str(func)
    return n.strip("_") if n.startswith("__") and n not in ("__getitem__", "__iter__", "__get__", "__set__", "__setitem__") and n.strip("_") not in SKIP else n

def short(v):
    if isinstance(v, str): return repr(v)
    if isinstance(v, (int, float, bool)) or v is None: return str(v)
    if isinstance(v, (list, tuple)) and all(isinstance(a, (int, float)) for a in v) and len(v) < 6: return str(tuple(v))
    return None

def kind(name, cls):
    n, c = name.split(".")[-1].lower(), cls.lower()
    if n.isdigit() and "." in name: n = name.split(".")[-2].lower()
    if "norm" in c: return "norm"
    if "embedding" in c or any(s in n for s in ("emb", "embed", "token", "pos", "fourier")): return "embed"
    if "head" in n: return "head"
    if "attn" in n or "attention" in c or n in ("qkv", "q", "k", "v", "o", "kv", "pool_q", "pool_k", "out_proj"): return "attn"
    if "mlp" in n or "ffn" in n or "feedforward" in c: return "mlp"
    if any(s in n for s in ("pool", "readout", "read", "cls", "to_emb", "flatten", "agg")) or "flatten" in c or "pool" in c: return "pool"
    if "conv" in c or n.startswith("up") or "upsample" in c: return "conv"
    if c in ("relu", "gelu", "silu", "sigmoid", "tanh", "leakyrelu", "dropout", "identity", "softmax", "elu", "mish"): return "act"
    if "block" in c or "layer" in c and "norm" not in c: return "block"
    if "encoder" in c or "enc" in n: return "enc"
    if "decoder" in c or "dec" in n: return "dec"
    if c == "linear": return "proj"
    return "module"

def op_kind(name):
    if name in ("scaled_dot_product_attention", "softmax", "einsum", "matmul", "bmm", "baddbmm", "rmatmul"): return "attn"
    if name in ("mean", "sum", "amax", "max", "min", "logsumexp", "std", "var", "norm"): return "pool"
    if name in ("relu", "gelu", "silu", "sigmoid", "tanh", "leaky_relu", "dropout", "elu", "softplus", "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos"): return "act"
    if "norm" in name: return "norm"
    if "conv" in name or "interpolate" in name: return "conv"
    if name in ("linear",): return "proj"
    return "op"

def hyper(m):
    """compact hyperparameters of a leaf module: 'Linear 128→384'"""
    g = lambda a: getattr(m, a, None)
    c = type(m).__name__
    if g("in_features") is not None: return f"{c} {m.in_features}→{m.out_features}"
    if g("in_channels") is not None:
        k, s = g("kernel_size"), g("stride")
        k, s = (k[0] if isinstance(k, tuple) and len(set(k)) == 1 else k), (s[0] if isinstance(s, tuple) and len(set(s)) == 1 else s)
        return f"{c} {m.in_channels}→{m.out_channels} k{k}" + (f" s{s}" if s not in (1, None) else "")
    if g("normalized_shape") is not None: return f"{c} {tuple(m.normalized_shape)[0] if len(m.normalized_shape) == 1 else tuple(m.normalized_shape)}"
    if g("num_features") is not None: return f"{c} {m.num_features}"
    if g("num_embeddings") is not None: return f"{c} {m.num_embeddings}×{m.embedding_dim}"
    if g("embed_dim") is not None and g("num_heads") is not None: return f"{c} d{m.embed_dim} h{m.num_heads}"
    if c == "Dropout": return f"Dropout p{m.p}"
    if c == "Flatten": return f"Flatten {m.start_dim}…{m.end_dim}"
    return c

def trace(model, batch, B):
    qual = {m: n for n, m in model.named_modules()}
    pmap = {id(t): "p:" + n for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    pshape = {"p:" + n: list(t.shape) for n, t in list(model.named_parameters()) + list(model.named_buffers())}
    pshape["__buffers__"] = {n for n, _ in model.named_buffers()}
    nodes, stack, prod = {}, [None], {}
    cur = [None, None]                                                                 # (python frame, fn node): ops of one helper-function call are grouped
    me = str(Path(__file__).resolve())

    def src(t): return prod.get(id(t), pmap.get(id(t)))
    def setprod(t, v):
        if id(t) not in prod: weakref.finalize(t, prod.pop, id(t), None)              # forget it when the tensor dies (ids get reused)
        prod[id(t)] = v
    def new(parent=-1, **kw):
        n = dict(id=len(nodes), parent=stack[-1] if parent == -1 else parent, items=[], **kw)
        nodes[n["id"]] = n
        if n["parent"] is not None: nodes[n["parent"]]["items"].append(n["id"])
        return n

    def pre(m, a, kw):
        cur[:] = None, None
        n = new(t="mod", mod=qual[m], cls=type(m).__name__, ins=[(src(t), list(t.shape)) for t in flat(a) + flat(kw)], outs=[], outsrc=[])
        stack.append(n["id"])
        if len(stack) == 2:                                                             # the model itself: its batch dict -> input nodes
            for k, v in (a[0].items() if a and isinstance(a[0], dict) else []):
                if isinstance(v, torch.Tensor): setprod(v, new(t="in", label=k, ins=[], outs=[list(v.shape)], dtype=str(v.dtype).replace("torch.", ""))["id"])

    def post(m, a, kw, o):
        n, cur[:] = nodes[stack[-1]], (None, None)
        items = o.items() if isinstance(o, dict) else [(None, v) for v in (o if isinstance(o, (list, tuple)) else [o])]
        for k, v in items:
            ts = flat(v)
            if not ts: continue
            if len(stack) == 2 and k is not None:                                     # the model's output dict -> output nodes
                new(t="out", label=k, ins=[(src(ts[0]), list(ts[0].shape))], outs=[list(ts[0].shape)])
            n["outsrc"] += [(src(t), list(t.shape)) for t in ts]
        stack.pop()
        for t in flat(o): n["outs"].append(list(t.shape)); setprod(t, n["id"])

    class Rec(TorchFunctionMode):
        def __torch_function__(self, func, types, a=(), kw=None):
            kw = kw or {}
            o = func(*a, **kw)
            try: record(func, a, kw, o)
            except Exception: pass
            return o

    def record(func, a, kw, o):
        name = fname(func)
        parent = stack[-1]
        if name in SKIP or parent is None: return
        outs, ins = flat(o), [t for t in flat(a) + flat(kw)]
        if not outs: return
        if not ins: return                                                              # torch.arange / zeros / ...: constants, not dataflow
        if name in CONSTANT: return                                                     # x.new_ones(...) (all-ones masks etc.): constants too
        if not qual_children[nodes[parent]["mod"]]: return                             # inside a leaf module (Linear's F.linear, ...): its card covers it
        srcs = [src(t) for t in ins]
        scalar_arith = name in BINARY and len(ins) == 1
        if name in TRANSPARENT or scalar_arith:                                         # shape-only / scalar ops: the edge carries the shape, no node
            for t in outs:
                if srcs[0] is not None: setprod(t, srcs[0])
            return
        detail = [s for s in (short(v) for v in a if not isinstance(v, torch.Tensor)) if s] + [f"{k}={short(v)}" for k, v in kw.items() if short(v) is not None and not isinstance(v, torch.Tensor)]
        fr = sys._getframe(2)
        while fr is not None and ("/torch/" in fr.f_code.co_filename or fr.f_code.co_filename == me): fr = fr.f_back
        if fr is not None and fr.f_code.co_name not in ("forward", "__call__", "_call_impl", "<module>"):  # a helper function (rotate(), normalize(), ...) -> its own group
            if cur[0] is not fr or nodes[cur[1]]["parent"] != parent:
                fn = getattr(fr.f_globals.get(fr.f_code.co_name), "__doc__", None) or ""
                cur[:] = fr, new(t="fn", name=fr.f_code.co_name, doc=fn.strip().split("\n")[0][:160], ins=[], outs=[], outsrc=[], file=f"{Path(fr.f_code.co_filename).name}:{fr.f_code.co_firstlineno}")["id"]
            parent = cur[1]
        else: cur[:] = None, None
        n = new(parent=parent, t="op", name=name, ins=[(s, list(t.shape)) for s, t in zip(srcs, ins)], outs=[list(t.shape) for t in outs], detail=", ".join(detail)[:60])
        for t in outs: setprod(t, n["id"])
        if nodes[parent]["t"] == "fn": nodes[parent]["outs"] = n["outs"]

    qual_children = {n: [c for c, _ in m.named_children()] for n, m in model.named_modules()}
    hs = [m.register_forward_pre_hook(pre, with_kwargs=True) for m in model.modules()] + [m.register_forward_hook(post, with_kwargs=True) for m in model.modules()]
    with torch.no_grad(), Rec(): model(batch)
    for h in hs: h.remove()
    return nodes, pshape

#*** stages: the default, paper-figure view

# optional nice names per arch file: (regex on the unit name, stage name, what it does). Unit names: module paths ("encoder.freq_enc.freqs" for
# a x N group), "<module path>.<op>" for tensor ops ("encoder.sub"), "<module path>.<fn>()" for helper functions. Rules are matched in forward
# order: a unit takes the first rule at or after the last rule used, so list them in call order; consecutive units with the same stage name form
# one stage; units no rule matches are grouped automatically (chains, parallel same-class siblings).
STAGES = {
    "transformer4": [
        (r"^encoder\.(mean|std|sub|truediv|div)$", "Normalize", "z-score each sample over lasers, bins and channels"),
        (r"^encoder\.embed$", "Tokenize", "cut each laser's spectrum into bands; embed each band"),
        (r"^encoder\.freq_enc\.freqs$", "Frequency encoder", "self-attention across the bands of each laser · 1D RoPE"),
        (r"^encoder\.freq_enc\.", "Merge bands", "MergeBlock: n_bands tokens → 1 per laser"),
        (r"^encoder\.laser_enc\.lasers$", "Laser encoder", "self-attention across the lasers · 2D RoPE"),
        (r"^encoder\.laser_enc\.", "Merge lasers", "MergeBlock: L lasers → latent D_l, + mask embedding"),
        (r"^(object|box)_head", "Aux heads", "classify object and box from the latent"),
        (r"^(decoder|sigmoid)$", "Mask decoder", "upsample the latent to a mask"),
    ],
    "transformer": [
        (r"normalize", "Normalize", "per-box standardization with train-set statistics"),
        (r"^encoder\.(laser_pos|add)$", "Laser embedding", "learned laser identity + 2D wall position"),
        (r"^encoder\.freq_enc\.(pad|embed|freq_band_emb|box_emb|add)$", "Tokenize", "cut each laser's spectrum into bands; add band, laser and box embeddings"),
        (r"^encoder\.freq_enc\.freqs$", "Frequency encoder", "self-attention across the bands of each laser"),
        (r"^encoder\.freq_enc\.", "Band pooling", "attention pooling over bands: one token per laser"),
        (r"^encoder\.laser_enc\.lasers$", "Laser encoder", "self-attention across the lasers"),
        (r"^encoder\.laser_enc\.", "Flatten", "all laser tokens into one embedding"),
        (r"^(decoder|sigmoid)$", "Mask decoder", "upsample the embedding to mask + outline"),
        (r"_head$|^sigmoid$", "Aux heads", "object, count, centre, area, box and orientation"),
    ],
    "hybrid": [
        (r"normalize", "Normalize", "per-box standardization with train-set statistics"),
        (r"^encoder\.(cat|conv\d)$", "Conv frontend", "3 stride-4 convs along frequency: 19 bands per laser"),
        (r"^encoder\.(embed|laser_pos|band_emb|box_emb|add)$", "Tokenize", "project to tokens; add laser, band and box embeddings"),
        (r"^encoder\.(pool_\w+|einsum|softmax|mean|mul)$", "Band pooling", "attention pooling over bands: one token per laser"),
        (r"^encoder\.lasers$", "Laser encoder", "self-attention across the lasers"),
        (r"^encoder\.to_emb$", "Flatten", "all laser tokens into one embedding"),
        (r"^(decoder|sigmoid)$", "Mask decoder", "upsample the embedding to mask + outline"),
        (r"head|^aux$|^sigmoid$", "Aux heads", "object, count, centre, area, box and orientation"),
    ],
}
STAGES["transformer_calib"] = STAGES["transformer"]
# helper functions shown as one named step: (regex on the enclosing module path, function name, label, what it does). Parallel calls of one
# function in the same block (rotate(q), rotate(k)) are always merged into one node.
FN_NAMES = {"transformer4": [(r"freq_enc", "rotate", "RoPE on q, k", "1D RoPE over band index (n_bands = 18)"),
                             (r"laser_enc", "rotate", "RoPE on q, k", "2D RoPE over laser grid (rows × cols): half the pairs encode the row, half the column")]}

# plain-English one-liners for an arch's own module classes (else: the class docstring's first line)
MODULE_WHAT = {"transformer4": {"MergeBlock": "merges N tokens into one: layer-norm each, its own learned matrix per token, summed (dropped tokens skipped)",
                                "Block": "self-attention (with rotary positions) + MLP, both residual"}}
LATENT_WHAT = {"transformer4": "one vector per sample: all the decoder and the heads see"}

# the main view: section columns, left -> right: (title, one plain sentence, stage names in it). Generic fallback in make_sections.
SECTIONS = {
    "transformer4": [
        ("Input", "Each laser's spectrum is rescaled, cut into frequency bands, and every band becomes a token of D numbers.", ["Normalize", "Tokenize"]),
        ("Frequency encoder", "Within each laser, the bands attend to each other, then are merged into one token per laser.", ["Frequency encoder ×4", "Merge bands"]),
        ("Laser encoder", "The lasers attend to each other, then are merged into one latent vector for the whole sample.", ["Laser encoder ×8", "Merge lasers", "latent"]),
        ("Decoder", "The latent vector is upsampled into a mask of where the objects are; two small heads guess the object and the box.", ["Mask decoder", "Aux heads"]),
    ],
}

def make_sections(nodes, root):
    """columns of the main view: [{title, what, items}] (items: root item ids in order)"""
    items = [nodes[i] for i in root["items"]]
    st = [n for n in items if n["t"] in ("stage", "latent")]
    by_label = {n["label"]: n["id"] for n in st}
    rules = SECTIONS.get(arch.stem)
    if rules and all(l in by_label for _, _, ls in rules for l in ls):
        secs = [dict(title=t, what=w, items=[by_label[l] for l in ls]) for t, w, ls in rules]
    else:
        first_rep = next((k for k, n in enumerate(st) if n.get("n")), None)
        lat = next((k for k, n in enumerate(st) if n["t"] == "latent"), None)
        if first_rep is None: secs = [dict(title=n["label"], what="", items=[n["id"]]) for n in st]
        else:
            end = lat + 1 if lat is not None else next((k for k in range(first_rep, len(st)) if not st[k].get("n") and k + 1 < len(st) and not st[k + 1].get("n")), len(st) - 1) + 1
            parts = [("Input", "Prepare the raw input.", st[:1] if first_rep > 1 else []), ("Tokenize", "Turn the input into tokens.", st[1 if first_rep > 1 else 0:first_rep]),
                     ("Encoder", "Mix the tokens with attention and pool them.", st[first_rep:end]), ("Decoder", "Produce the outputs.", st[end:])]
            secs = [dict(title=t, what=w, items=[n["id"] for n in ns]) for t, w, ns in parts if ns]
    ins = [n["id"] for n in items if n["t"] == "in"]
    secs[0]["items"] = ins + secs[0]["items"]
    owner = {i: k for k, sc in enumerate(secs) for i in sc["items"]}
    for n in items:                                                                    # outputs go with the stage that makes them
        if n["t"] == "out":
            src_ = next((e[0] for e in root["edges"] if e[1] == n["id"]), None)
            secs[owner.get(src_, len(secs) - 1)]["items"].append(n["id"])
    return secs

def human(name): return name.replace("_", " ").strip().capitalize() or "Model"

def make_stages(nodes, root, anc, mods_):
    """flatten the call tree along the path to the x N blocks into units, then group consecutive units into stages under the root"""
    def has_rep(i): n = nodes[i]; return n["t"] == "rep" or (n["t"] == "mod" and any(has_rep(c) for c in n["items"]))
    units, split = [], set()
    def collect(i):
        for c in list(nodes[i]["items"]):
            n = nodes[c]
            if n["t"] in ("in", "out"): continue
            if n["t"] == "mod" and any(has_rep(x) for x in n["items"]): split.add(c); collect(c)   # a wrapper around the repeated blocks: open it up
            else: units.append(c)
    collect(root["id"])
    def inner(s_, shape):                                                              # an opened wrapper's output -> whatever made it inside
        while isinstance(s_, int) and s_ in split:
            cands = nodes[s_]["outsrc"]
            s_ = next((a for a, b in cands if b == shape), cands[0][0] if cands else None)
        return s_
    for n in nodes.values():
        n["ins"] = [(inner(a, b), b) for a, b in n["ins"]]
        if "outsrc" in n: n["outsrc"] = [(inner(a, b), b) for a, b in n["outsrc"]]
    def uname(u):
        n, pm = nodes[u], nodes[nodes[u]["parent"]].get("mod", "")
        if n["t"] == "rep": return re.sub(r"\.\d+$", "", n["mod"])
        if n["t"] == "mod": return n["mod"]
        return (pm + "." if pm else "") + (n["name"] + "()" if n["t"] == "fn" else n["name"])
    rules = STAGES.get(arch.stem, [])
    last = [0]
    def match(nm):
        for k in range(last[0], len(rules)):
            if re.search(rules[k][0], nm): last[0] = k; return rules[k][1:]
    stages = []
    for u in units:
        n, lab, par = nodes[u], match(uname(u)), nodes[u]["parent"]
        isop, prev = n["t"] in ("op", "fn"), stages[-1] if stages else None
        same = prev is not None and prev["par"] == par
        consumes = prev is not None and any(isinstance(s, int) and set(anc(s)) & set(prev["units"]) for s, _ in n["ins"])
        if prev and lab and prev["lab"] and prev["lab"][0] == lab[0]: join = True
        elif prev and prev["opsonly"] and not prev["lab"] and same: join = True; prev["lab"] = lab
        elif lab or (prev and prev["lab"]): join = bool(prev and not lab and isop and same and not prev["rep"])
        elif isop: join = bool(prev and not prev["rep"] and (same or consumes))
        else:
            first = nodes[prev["units"][0]] if prev else None
            parallel = bool(prev and first["t"] == "mod" and n["t"] == "mod" and first["cls"] == n["cls"] and same and {s for s, _ in first["ins"]} == {s for s, _ in n["ins"]})
            join = bool(prev and same and not prev["rep"] and n["t"] != "rep" and (consumes or parallel))     # a chain, or parallel siblings of one class (heads)
        if join: prev["units"].append(u); prev["opsonly"] &= isop; prev["rep"] = False
        else: stages.append(dict(units=[u], lab=lab, par=par, opsonly=isop, rep=n["t"] == "rep"))
    for st in stages:
        us = [nodes[u] for u in st["units"]]
        reps = [x for x in us if x["t"] == "rep"]
        nrep = len(reps[0]["members"]) if len(reps) == 1 and all(x["t"] in ("rep", "op", "fn") for x in us) else 0
        if st["lab"]: label, what = st["lab"]
        elif reps: label, what = human(uname(reps[0]["id"]).split(".")[-1]), ""
        else:
            ms = sorted([x for x in us if x["t"] in ("mod", "rep")], key=lambda x: -sum(q.numel() for q in mods_[x["mod"]].parameters()))
            names = [x["mod"].split(".")[-1] for x in ms]
            label = ("Heads" if len(names) > 1 and all(x.endswith("head") for x in names) else " + ".join(human(x) for x in names[:2])) if ms else human(us[0]["name"])
            what = ", ".join(dict.fromkeys(x["mod"].split(".")[-1] if x["t"] in ("mod", "rep") else x["name"] for x in us))[:70]
        if nrep and "×" not in label: label += f" ×{nrep}"
        g = dict(id=max(nodes) + 1, parent=root["id"], t="stage", items=list(st["units"]), ins=[], outs=us[-1]["outs"], outsrc=[], label=label, what=what, n=nrep, rep=reps[0]["id"] if reps else None)
        nodes[g["id"]] = g
        for u in st["units"]:
            nodes[u]["parent"] = g["id"]
            for m_ in nodes[u].get("members", []): nodes[m_]["parent"] = g["id"]          # x N members live with their group
        st["id"] = g["id"]
    ios = [i for i in root["items"] if nodes[i]["t"] in ("in", "out")]
    root["items"] = [i for i in ios if nodes[i]["t"] == "in"] + [st["id"] for st in stages] + [i for i in ios if nodes[i]["t"] == "out"]
    return stages

ACT = {"ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU", "Mish", "Softplus"}
def describe(m, n, nodes):
    """one plain-English line for a module card"""
    c, g = type(m).__name__, lambda a: getattr(m, a, None)
    k = g("kernel_size"); k = k[0] if isinstance(k, tuple) else k
    s = g("stride"); s = s[0] if isinstance(s, tuple) else s
    kids = list(m.children())
    if not type(m).__module__.startswith("torch."): c = "_custom"                     # own classes: described by their docstring below
    if c == "Linear": return f"linear {m.in_features} → {m.out_features}"
    if c in ("LayerNorm", "RMSNorm"): return f"{'layer' if c == 'LayerNorm' else 'RMS'} norm over {tuple(m.normalized_shape)[-1]} features"
    if c.startswith("BatchNorm"): return f"batch norm, {m.num_features} channels"
    if c.startswith("ConvTranspose"): return f"{k}×{k} transposed conv, ×{s} upsample, {m.in_channels} → {m.out_channels} ch"
    if c.startswith("Conv"): return f"{k}×{k} conv, {m.in_channels} → {m.out_channels} ch" + (f", stride {s}" if s not in (1, None) else "")
    if c == "Embedding": return f"learned table, {m.num_embeddings} × {m.embedding_dim}"
    if c in ACT: return "activation"
    if c == "Dropout": return f"dropout {m.p}"
    if c == "Flatten": return "flatten"
    if kids and all(not list(x.children()) for x in kids) and type(m).__module__.startswith("torch."):   # Sequential of leaves
        lins = [x for x in kids if type(x).__name__ == "Linear"]
        if lins and all(type(x).__name__ in ACT | {"Linear", "Dropout"} for x in kids):
            return "MLP " + " → ".join([str(lins[0].in_features)] + [str(x.out_features) for x in lins])
        return " → ".join(type(x).__name__ for x in kids)
    if type(m).__name__ in MODULE_WHAT.get(arch.stem, {}): return MODULE_WHAT[arch.stem][type(m).__name__]
    doc = (type(m).__doc__ or "").strip().split("\n")[0] if not type(m).__module__.startswith("torch.") else ""   # own docstring only (not nn.Module's)
    return doc[:90] or (", ".join(n_ for n_, _ in m.named_children())[:70] if kids else type(m).__name__)

SECS = [None]
def build_graph(model, nodes, pshape):
    """per container: its items (children, x N groups, param nodes) and the edges between them, resolved through the call tree"""
    mods = dict(model.named_modules())
    total = sum(p.numel() for p in model.parameters())
    group_of, groups = {}, []
    for n in list(nodes.values()):                                                     # a helper-function group with one op: just the op
        if n["t"] == "fn" and len(n["items"]) == 1:
            c = nodes[n["items"][0]]; c["parent"] = n["parent"]
            its = nodes[n["parent"]]["items"]; its[its.index(n["id"])] = c["id"]; del nodes[n["id"]]
    def anc(i):
        while i is not None: yield i; i = nodes[i]["parent"]
    for p in list(nodes.values()):                                                     # parallel calls of one helper (rotate(q), rotate(k)) -> one node
        by = {}
        for i in p["items"]:
            if nodes[i]["t"] == "fn": by.setdefault(nodes[i]["name"], []).append(i)
        for ids in by.values():
            if len(ids) < 2: continue
            inner = {j: set(nodes[j]["items"]) for j in ids}
            if any(isinstance(s_, int) and s_ in inner[k] for j in ids for c in nodes[j]["items"] for s_, _ in nodes[c]["ins"] for k in ids if k != j): continue
            keep = ids[0]
            for j in ids[1:]:
                for c in nodes[j]["items"]: nodes[c]["parent"] = keep; nodes[keep]["items"].append(c)
                p["items"].remove(j); del nodes[j]
            nodes[keep]["merged"] = len(ids)
    # x N: consecutive calls of <prefix>.<i> modules, same class, same in / out shapes
    for n in list(nodes.values()):
        run = []
        for i in n["items"] + [None]:
            c = nodes[i] if i is not None else None
            sig = c and c["t"] == "mod" and re.match(r"(.*)\.(\d+)$", c["mod"]) and (re.match(r"(.*)\.(\d+)$", c["mod"])[1], c["cls"], str([s for _, s in c["ins"]]), str(c["outs"]))
            if run and sig and sig == run[0][1]: run.append((i, sig)); continue
            if len(run) > 1: groups.append((n["id"], [r[0] for r in run]))
            run = [(i, sig)] if sig else []
    for parent, members in groups:
        g = dict(id=max(nodes) + 1, parent=parent, t="rep", members=members, tpl=members[0], items=[], ins=nodes[members[0]]["ins"], outs=nodes[members[-1]]["outs"],
                 mod=nodes[members[0]]["mod"], cls=nodes[members[0]]["cls"], outsrc=[])
        nodes[g["id"]] = g
        its = nodes[parent]["items"]
        k = its.index(members[0]); its[k:k + len(members)] = [g["id"]]
        for mbr in members: group_of[mbr] = g["id"]

    root = next(n for n in nodes.values() if n["t"] == "mod" and n["mod"] == "")
    stages = make_stages(nodes, root, anc, mods)
    for g in [n for n in nodes.values() if n["t"] in ("fn", "stage")]:                 # fn / stage groups: inputs / outputs = what crosses the group boundary
        g["ins"], g["outsrc"] = [], []
    groups_ = [n for n in nodes.values() if n["t"] in ("fn", "stage")]
    for c in list(nodes.values()):
        if c["t"] in ("stage",): continue
        a_c = set(anc(c["id"]))
        for s_, shape in c["ins"]:
            a_s = set(anc(s_)) if isinstance(s_, int) else set()
            for g in groups_:
                if g["id"] in a_s and g["id"] not in a_c: g["outsrc"].append((s_, shape))
                if g["id"] in a_c and g["id"] not in a_s and not (g["t"] == "stage" and isinstance(s_, str)): g["ins"].append((s_, shape))
    for g in groups_:                                                                  # dedupe
        g["ins"] = list({(str(a), str(b)): (a, b) for a, b in g["ins"]}.values()); g["outsrc"] = list({(str(a), str(b)): (a, b) for a, b in g["outsrc"]}.values())

    def up(i):                                                                         # node id -> its chain of ancestors (self first)
        while i is not None: yield i; i = nodes[i]["parent"]

    params = {}
    def resolve(s, C):
        if s is None: return None
        if isinstance(s, str):                                                         # a parameter / buffer consumed directly in C's own code
            # (also when C got it as an argument: a pill beside its consumer reads better than a long wire from the top)
            if (C, s) not in params:
                nm = s[2:]
                pn = dict(id=max(nodes) + 1, parent=C, t="param", label=nm.split(".")[-1], full=nm, outs=[pshape[s]], ins=[], items=[], buffer=nm in pshape["__buffers__"])
                nodes[pn["id"]] = pn; nodes[C]["items"].insert(0, pn["id"]); params[(C, s)] = pn["id"]
            return params[(C, s)]
        for a in up(s):
            if nodes[a]["parent"] == C or (a in group_of and nodes[group_of[a]]["parent"] == C): return group_of.get(a, a)
        return "IN"

    for n in list(nodes.values()):
        if not n["items"] and n["t"] != "rep": continue
        if n["t"] == "rep": continue
        edges = []
        for i in list(n["items"]):
            c = nodes[i]
            ins = c["ins"]
            if c["t"] == "rep":                                                         # external inputs of any member enter the group, once
                ins = list({(str(x[0]), str(x[1])): x for m in c["members"] for x in nodes[m]["ins"]}.values())
            for s, shape in ins:
                f = resolve(s, n["id"])
                if f is not None and f != i: edges.append([f, i, shape])
        for s, shape in (n.get("outsrc", []) if n["mod" if "mod" in n else "t"] != "" else []):  # the model's own outputs are the out nodes
            f = resolve(s, n["id"])
            if f is not None: edges.append([f, "OUT", shape])
        seen, n["edges"] = {}, []
        for e in edges:                                                                 # dedupe (q, k, v from one qkv -> one edge "x3")
            key = (e[0], e[1], str(e[2]))
            if key in seen: n["edges"][seen[key]][3] += 1
            else: seen[key] = len(n["edges"]); n["edges"].append(e + [1])

    fan = {}                                                                            # a stage output that feeds several stages -> its own "latent" node
    for e in root["edges"]:
        if nodes[e[1]]["t"] == "stage": fan.setdefault((e[0], str(e[2])), []).append(e)
    for (src_, _), es in fan.items():
        if len(es) < 2 or nodes[src_]["t"] != "stage": continue
        shape = es[0][2]
        outs_ = [e for e in root["edges"] if e[0] == src_ and nodes[e[1]]["t"] == "out" and e[2] == shape]
        name = nodes[outs_[0][1]]["label"] if outs_ else "latent"
        lat = dict(id=max(nodes) + 1, parent=root["id"], t="latent", label=name, items=[], ins=[(src_, shape)], outs=[shape], returned=bool(outs_),
                   what=LATENT_WHAT.get(arch.stem, "one vector per sample, shared by " + " and ".join(nodes[e[1]]["label"].lower() for e in es)))
        nodes[lat["id"]] = lat
        root["items"].insert(root["items"].index(src_) + 1, lat["id"])
        for e in es: e[0] = lat["id"]
        for e in outs_: root["edges"].remove(e); root["items"].remove(e[1]); del nodes[e[1]]
        root["edges"].append([src_, lat["id"], shape, 1])
    used = {e[0] for e in root["edges"]}
    for i in [i for i in root["items"] if nodes[i]["t"] == "in" and i not in used]: root["items"].remove(i); del nodes[i]
    SECS[0] = make_sections(nodes, root)
    for i in [i for i in root["items"] if nodes[i]["t"] == "in" and i not in used]: root["items"].remove(i); del nodes[i]   # batch keys the forward never reads
    out = {}
    for n in nodes.values():
        d = {k: n[k] for k in ("id", "t", "items", "edges", "members", "tpl", "label", "detail", "dtype", "full") if k in n}
        d["ins"], d["outs"] = [s for _, s in n["ins"]], n["outs"]
        if n["t"] == "latent":
            d |= dict(what=n["what"], returned=n["returned"])
        elif n["t"] == "stage":
            d |= dict(label=n["label"], what=n["what"], kind="stage", n=n["n"], rep=n["rep"], params=0)
        elif n["t"] in ("mod", "rep"):
            m = mods[n["mod"]]
            own, tot = sum(q.numel() for q in m.parameters(recurse=False)), sum(q.numel() for q in m.parameters())
            mult = len(n["members"]) if n["t"] == "rep" else 1
            parts = n["mod"].split(".")
            label = (f"{parts[-2]}[{parts[-1]}]" if len(parts) > 1 and parts[-1].isdigit() else parts[-1]) or type(m).__name__
            own_cls = not type(m).__module__.startswith("torch.") and n["mod"] != ""
            if own_cls and n["t"] == "mod" and not parts[-1].isdigit(): label = type(m).__name__   # the arch's own blocks by class name (MergeBlock, Up, ...)
            if n["t"] == "rep": label = f"{parts[-2]}[{parts[-1]}…{nodes[n['members'][-1]]['mod'].split('.')[-1]}]"
            leaf = not list(m.children())
            only_mods = all(nodes[i]["t"] in ("mod", "param") for i in nodes[n["members"][0] if n["t"] == "rep" else n["id"]]["items"])
            chain = "" if not only_mods else " · ".join(hyper(c) for c in m.children()) if not leaf and all(not list(c.children()) for c in m.children()) and len(list(m.children())) <= 6 else ""
            d |= dict(label=label, cls=type(m).__name__, mod=n["mod"], kind=kind(n["mod"] or "model", type(m).__name__), params=tot * mult, own=own, n=mult,
                      trainable=all(q.requires_grad for q in m.parameters()) if tot else None, extra=m.extra_repr(), sub=hyper(m) if leaf else (chain or type(m).__name__),
                      leafonly=bool(chain), what=describe(m, n, nodes), pct=100 * tot * mult / max(total, 1), doc=(inspect.getdoc(type(m)) or "").split("\n")[0][:160] if type(m).__module__ not in ("torch.nn.modules.linear",) and not type(m).__module__.startswith("torch.") else "")
        elif n["t"] == "op":
            d |= dict(label=SYMBOL.get(n["name"], OPNAME.get(n["name"], n["name"])), name=n["name"], kind=op_kind(n["name"]), sym=n["name"] in SYMBOL)
        elif n["t"] == "fn":
            path = next((nodes[a]["mod"] for a in anc(n["id"]) if nodes[a].get("mod")), "")
            ov = next(((lab, w) for rx, fn_, lab, w in FN_NAMES.get(arch.stem, []) if fn_ == n["name"] and re.search(rx, path)), None)
            label = ov[0] if ov else n["name"] + "()" + (f" ×{n['merged']}" if n.get("merged") else "")
            d |= dict(label=label, name=n["name"], kind="fn", doc=n["doc"], file=n["file"], sub=f"{len(n['items'])} ops", semantic=bool(ov),
                      what=ov[1] if ov else n["doc"] or f"helper function, {len(n['items'])} ops")
        elif n["t"] == "param":
            d |= dict(kind="param", params=math.prod(n["outs"][0]), buffer=n["buffer"])
        else:
            d |= dict(kind="io")
        out[n["id"]] = d
    def psum(i):
        d = out[i]
        return d.get("params", 0) if d["t"] in ("mod", "rep", "param") else sum(psum(c) for c in d["items"])
    for i, d in out.items():
        if d["t"] == "stage": d["params"] = sum(psum(c) for c in d["items"] if not out[c].get("buffer")); d["pct"] = 100 * d["params"] / max(total, 1)
    return out

ALIASES = {"D": ("D", "d", "dim", "hidden", "hidden_dim"), "D_l": ("LATENT", "latent_dim", "d_model", "latent"), "H": ("HEADS", "heads", "n_heads", "num_heads"),
           "BINS": ("BINS", "bins_per_band")}
def dim_names(info, B, kwargs, sig_defaults, consts, model, H=64, W=64):
    cand = {"mask_h": H, "mask_w": W, "B": B, "L": info["n_laser_rows"] * info["n_laser_cols"], "F": info["n_freqs"], "C": info["n_channels"], "rows": info["n_laser_rows"], "cols": info["n_laser_cols"],
            "K": max(info["n_objects"]), "n_boxes": len(info["boxes"]), "n_obj_cls": len(info["objects"]) - 1}
    for k, v in list(sig_defaults.items()) + list(kwargs.items()) + list(consts.items()):
        if isinstance(v, int) and not isinstance(v, bool) and v > 1: cand[k] = v
    for m in model.modules():
        if type(m).__module__.startswith("torch."): continue
        for k, v in vars(m).items():
            if not k.startswith("_") and k not in ("training",) and len(k) > 1 and isinstance(v, int) and not isinstance(v, bool) and v > 1: cand.setdefault(k, v)
    cand = {k: v for k, v in cand.items() if "layer" not in k.lower()}             # layer counts are never tensor dims
    if any(o in cand for o in ALIASES["H"]):                                       # H = attention heads; a mask-size constant H / W is mask_h / mask_w
        for k in ("H", "W"): cand.pop(k, None)
    for new, olds in ALIASES.items():
        for o in olds:
            if o in cand: cand.setdefault(new, cand[o])
        for o in olds:
            if o != new: cand.pop(o, None)
    if "D" in cand and "H" in cand: cand["D_h"] = cand["D"] // cand["H"]               # head dim
    if "BINS" in cand and "D" in cand and cand["D"] != cand["BINS"] * cand["C"]: cand["BINS·C"] = cand["BINS"] * cand["C"]   # one band token's raw values
    if "n_bands" in cand: cand["L+n_bands"] = cand["L"] + cand["n_bands"]          # a laser mask and a band mask, concatenated
    cand["B·L"] = B * cand["L"]
    by = {}
    for k, v in cand.items(): by.setdefault(v, []).append(k)
    return {str(v): "/".join(dict.fromkeys(ks)) for v, ks in by.items()}, cand

def build():
    DEFAULTS, AUGS, COMMON = run_defaults()
    key = args.model or arch.stem
    spec = importlib.util.spec_from_file_location(f"arch_diagram_{arch.stem}", arch)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    cls = find_model_class(mod)
    sig = inspect.signature(cls.__init__).parameters
    defaults = DEFAULTS.get(key, {})
    accepts_kw = any(q.kind == q.VAR_KEYWORD for q in sig.values())
    kwargs = {k: v for k, v in defaults.items() if k in sig or (accepts_kw and k not in COMMON and k not in ("window", "t_warmup", "max_duration", "checkpoint_interval", "microbatch"))}
    kwargs |= parse_knobs(knobs, {**{k: q.default for k, q in sig.items() if q.default is not inspect._empty}, **defaults})
    info, info_src = data_info()
    torch.manual_seed(0)
    model = cls(info, **kwargs).eval()
    H, W = kwargs.get("height", defaults.get("height", 64)), kwargs.get("width", defaults.get("width", 64))
    B = args.batch
    nodes, pshape = trace(model, dummy_batch(info, B, H, W), B)
    graph = build_graph(model, nodes, pshape)
    root = next(n["id"] for n in graph.values() if n["t"] == "mod" and n["mod"] == "")
    sig_defaults = {k: q.default for k, q in sig.items() if q.default is not inspect._empty}
    consts = {k: v for k, v in vars(mod).items() if k.isupper() and isinstance(v, int)}
    names, cand = dim_names(info, B, kwargs, sig_defaults, consts, model, H, W)
    n_params, n_train = sum(q.numel() for q in model.parameters()), sum(q.numel() for q in model.parameters() if q.requires_grad)
    doc = (mod.__doc__ or "").strip()
    meta = dict(cls=cls.__name__, arch=str(arch.relative_to(ROOT)) if arch.is_relative_to(ROOT) else str(arch), key=key, time=f"{datetime.now():%Y-%m-%d %H:%M}",
                mtime=f"{datetime.fromtimestamp(arch.stat().st_mtime):%Y-%m-%d %H:%M}", params=n_params, trainable=n_train,
                kwargs={k: v for k, v in kwargs.items()}, overrides=knobs, B=B, info_src=info_src, doc=doc,
                buffers=sum(b.numel() for b in model.buffers()), sig={k: sig_defaults[k] for k in sig_defaults if k not in kwargs})
    return dict(meta=meta, nodes=graph, root=root, dims=names, dimlist=cand, sections=SECS[0])

#*** 4 html

def write(page):
    out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_text(page); print(f"wrote {out_path}")

TEMPLATE = (Path(__file__).parent / "arch_diagram.html")
try:
    data = build()
    page = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, default=str)).replace("__TITLE__", html.escape(f"{data['meta']['cls']} architecture"))
    write(page)
except Exception:
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    write(TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(dict(error=tb, meta=dict(arch=str(arch), time=f"{datetime.now():%Y-%m-%d %H:%M}")))).replace("__TITLE__", "Architecture: error"))
    sys.exit(1)
