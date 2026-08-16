# The math behind the measurement, and what it implies for the architecture

## 1. What is actually being measured

The box wall is a linear elastic structure. Discretized, its motion obeys

    M x''(t) + C x'(t) + K x(t) = f(t)

with `M` mass, `C` damping, `K` stiffness, `x` the displacement at each point, `f` the acoustic
forcing from the speaker. Transforming to frequency (`d/dt -> i*omega`):

    (K + i*omega*C - omega^2*M) X(omega) = F(omega)

so

    X(omega) = H(omega) F(omega),    H(omega) = (K + i*omega*C - omega^2*M)^(-1)

`H` is the **transfer function**, or frequency response function. This is the object of interest:
it depends only on the box+object system, not on how hard you drive it.

Your measurement is `X` at 100 points on the wall. Since `F` (the chirp) is fixed and identical
across every sample, `X` is proportional to `H` and you are, up to a per-recording gain, measuring
`H` directly.

### Where the resonances come from

`H` blows up where `det(K + i*omega*C - omega^2*M) = 0`. Those are the **poles** -- the resonant
modes. Near a pole, `H` is dominated by that mode:

    H(omega) ~ sum_r  (phi_r phi_r^T) / (omega_r^2 - omega^2 + 2*i*zeta_r*omega_r*omega)

where `omega_r` is the r-th natural frequency, `zeta_r` its damping, and `phi_r` its **mode shape**
-- a standing-wave pattern over the structure with nodes (zero motion) and antinodes (maximum motion).

This is the modal superposition formula, and it is the key to everything below. Note the numerator
`phi_r phi_r^T`: the contribution of mode `r` at measurement point `j` is scaled by `phi_r(j)`.

## 2. Why it is NOT linear in the objects

Placing a cube in the box perturbs the mass and stiffness matrices:

    M -> M + dM_A     (cube A)
    M -> M + dM_B     (cube B)
    M -> M + dM_A + dM_B   (both)

The perturbations to `M` **are** additive. But the measurement is `H`, which is the **inverse**:

    H_A   = (K + i*omega*C - omega^2*(M + dM_A))^(-1)
    H_B   = (K + i*omega*C - omega^2*(M + dM_B))^(-1)
    H_AB  = (K + i*omega*C - omega^2*(M + dM_A + dM_B))^(-1)

and matrix inversion is not linear:

    H_AB != H_A + H_B - H_0

To see the size of the error, write `A_0 = K + i*omega*C - omega^2*M` and `E_X = -omega^2*dM_X`.
The Neumann/Woodbury expansion gives

    H_AB = H_0 - H_0(E_A + E_B)H_0 + H_0(E_A + E_B)H_0(E_A + E_B)H_0 - ...

The first-order term **is** additive: `-H_0 E_A H_0 - H_0 E_B H_0`. Superposition holds to first
order. But the second-order term contains the **cross terms**

    H_0 E_A H_0 E_B H_0  +  H_0 E_B H_0 E_A H_0

which describe cube A and cube B interacting *through the box* -- A shifts the modes, which changes
how B is coupled to them. These have no counterpart in `H_A` or `H_B` separately, so they are exactly
what a linear mixture cannot reproduce.

The expansion converges only when `||H_0 E|| < 1`, i.e. for small perturbations. A cube is not a
small perturbation: it visibly moves resonances. So the cross terms are not a correction, they are
first-class.

### Why this kills mixture-based data synthesis

You cannot build 3-object training data as `f(FFT_A, FFT_B, FFT_C)`, because:

1. **Inversion is nonlinear** (above). The cross terms are the physics of multi-object scenes.
2. **You store magnitude only.** Even if the complex responses were additive, `|a+b| != |a|+|b|`.
   Two spectra with equal magnitude and opposite phase cancel; magnitudes cannot express that.
3. **Anti-resonances are destroyed by summation.** These are the **zeros** of `H` -- frequencies
   where two modes' contributions cancel at a given measurement point. Notebook 57 found the dips
   carry *more* position information than the peaks (dips alone rho = -0.523). A zero is a
   cancellation, so it exists only in the exact sum; adding two spectra fills in each other's
   notches and erases precisely the most informative feature.
4. **You cannot even build the matched triples** to test it: 1-object and 2-object scenes sit ~40 px
   apart (half a grid pitch), and green-cube has no solo layout in gastronorm.

## 3. Phase: why it looked useless, and how to recover it

The phase of `H` is physical and informative -- it encodes the relative timing of the response, and
it flips by pi as you sweep through a resonance.

The problem is the **time origin**. Each recording triggers camera and audio independently, with
jitter of order ~1 ms. A time shift `tau` transforms the spectrum as

    Y(f) -> Y(f) exp(-2*pi*i*f*tau)

i.e. it adds `-2*pi*f*tau` to the phase -- a **linear ramp in frequency**. At f = 1000 Hz, tau = 1 ms
is a full 2*pi cycle, so the phase at the top of your band is completely randomized between samples.
This is why notebook 57 found phase made things worse and concluded it was unusable.

But an unknown additive linear ramp is a **gauge**, not noise. Differentiating with respect to `f`
removes it. Discretely, the difference between adjacent bins

    dphi(f) = arg( exp(i*(phi(f + df) - phi(f))) )

is invariant to `tau`, because the ramp contributes the same constant `-2*pi*df*tau` at every bin.
Up to that constant, `dphi/df` is the **group delay** -- a genuine physical property.

**Measured** (notebook 68). Circular resultant length `R` (0 = random, 1 = identical), across the 8
speakers at a fixed position:

| quantity | R |
|---|---|
| raw phase | 0.41 |
| group delay | **0.76** |

and across the 100 lasers within one sample: 0.22 -> **0.79**.

So phase is recoverable. Whether it is *useful* is a separate question, and the answer here is no:
as a feature it scores rho = +0.093 (or +0.318 magnitude-weighted), and concatenating it with
log-magnitude monotonically *hurts* (rho +0.396 -> +0.393 -> +0.375 -> +0.301 -> +0.171 as the phase
weight rises). Phase carries real information that is largely redundant with magnitude, and 247k
extra dimensions cost more than it is worth. **Encode phase as (cos, sin), never raw angles** --
phase is circular, and raw angles put a false discontinuity at +/-pi.

## 4. On the chirp, spectrograms, and stationarity

An earlier claim in this project's planning -- that "a global FFT assumes stationarity that a sweep
violates" -- was **wrong**, and the correction matters.

For a linear time-invariant system, the output spectrum is `Y(f) = H(f) X(f)` **regardless** of how
`X` distributes its energy in time. A chirp that sweeps 50 -> 1000 Hz has support across the whole
band, so `H(f) = Y(f)/X(f)` is recoverable from a single global FFT. Resonances are time-invariant
properties of the structure; time is a nuisance axis. **The global FFT is a valid estimator and the
intuition behind the sweep design is sound.**

The real limitation is **variance, not bias**. The gastronorm chirp is 1.0 s over 950 Hz, so each
frequency is driven for roughly

    1.0 s * (0.77 Hz / 950 Hz) ~ 0.8 ms

and never revisited. Capture is 1.3 s, so ~23% of the window contains no excitation at all and
contributes only noise. There is no repetition to average over, so every bin is a single-shot
estimate.

A spectrogram would not add information -- it would only let you *reject* the unexcited portion of
the window, a modest SNR gain. Welch-style averaging needs repeated realizations, which a single
sweep does not provide. **The fix is a capture-side change (repeat the sweep N times and average the
resulting spectra, which cuts noise variance by N), not a different transform of existing data.**

## 5. Why the current cross-attention decoder is "structurally per-pixel"

`AttnDecoder` (src/model/arch.py) creates one learned query per **output pixel**:

    self.query_seed = nn.Parameter(torch.zeros(1, out_h * out_w, d_model))   # 630 queries at 21x30

Each query cross-attends into the laser tokens and a shared `nn.Linear(d_model, 1)` maps its output
to that pixel's logit.

The DETR/MaskFormer idea is **set prediction**: a small number of queries (~100), each representing
a *candidate object*, matched to ground-truth objects by Hungarian assignment. The inductive bias is
that a query is an object -- it has a location, an extent, and an existence probability, and objects
compete for queries. That is what makes the count structure explicit.

Binding one query to one pixel discards all of it:

- A query no longer means "an object", it means "the pixel at (r, c)". There is nothing to match, so
  no Hungarian assignment and no set-prediction bias.
- The 630 queries are **independently parameterized**, so nothing ties neighbouring pixels together
  beyond what RoPE supplies. The model can represent an arbitrary per-pixel lookup.
- The output is still a single fused occupancy map. Nothing in the architecture represents "how many
  things are here", which is precisely the generalization axis of interest.
- Cost is `O(H*W)` queries -- the *expensive* part of DETR -- with none of the benefit.

Note also that MaskFormer's headline claim is often misread: it outperforms per-pixel baselines when
the number of **classes** is large (ADE20K-style large vocabularies). It makes no claim about
extrapolating to unseen instance *counts*.

### How to make it a real set-prediction decoder

    K queries (K ~ 8, K >> max objects, not 630)
      -> each cross-attends into the laser tokens
      -> each emits: a mask embedding e_k in R^d, and an existence logit p_k
    mask_k = sigmoid(e_k . pixel_embedding)         # low-rank, hence inherently smooth
    final  = max_k (p_k * mask_k)   or   sum_k

Train with Hungarian matching between the K predicted masks and the ground-truth instance masks.
Three properties follow that the current decoder cannot have:

1. **Count is explicit** -- it is `sum_k p_k`, supervised directly.
2. **Masks are low-rank** (a dot product between a query embedding and a pixel embedding), so they
   are smooth by construction rather than by regularization.
3. **Compositional capacity** -- a scene with 3 objects activates 3 queries. Nothing about the
   architecture changes between 2 and 3 objects, which is exactly the property missing today.

**Prerequisite:** Hungarian matching needs per-instance ground-truth masks. The current targets are
already-merged binary unions (`np.any(...)` at capture time, irreversible). But per-object boxes,
identities and COMs survive in `samples/*/metadata.jsonl`, and since the cubes are rigid and
constant-size (~96-105 px squares), instance masks are reconstructible from boxes alone -- no SAM3
rerun needed for a first pass. All 3008 `02_cropped_overhead.png` also survive if full
re-segmentation is wanted later. Target format: experiment-25's `smasks/all.npy`, a 0/1/2 instance
label map.

**Caveat worth stating plainly:** with only 40 three-object samples across 5 positions and 1 layout,
this architecture change cannot be *validated* on 3-object generalization. It should be judged on
1<->2 transfer and held-out positions, where there is statistical power, with 3-object kept as a
confirmatory readout only.
