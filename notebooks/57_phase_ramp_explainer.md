# Why phase doesn't work in this dataset: the linear ramp

Companion note to `57_similar_ffts.ipynb`. Written in plain text/ASCII math (no LaTeX) so it
renders anywhere.

---

## 1. What phase is

The FFT of a sample is **complex**. Each frequency bin holds two numbers, usually written as a
magnitude and an angle:

```
X(f) = |X(f)| * exp(i * phi(f))
```

- `|X(f)|` — **magnitude**: how much energy sits at frequency `f`.
- `phi(f)` — **phase**: the *timing* of that frequency component. Where in its cycle the
  sinusoid at frequency `f` happens to be at `t = 0`. Measured in radians; one full cycle is
  `2*pi ~= 6.283`.

Every metric in the notebook except `coherence` and `cpsd` calls `np.abs()`, which keeps
`|X(f)|` and throws `phi(f)` away. This note explains why that turns out to be the right call.

---

## 2. The shift theorem: where "ramp" comes from

Here is the one piece of math that drives everything. Take any signal `x(t)` and delay it by
`tau` seconds. In the frequency domain:

```
x(t - tau)   <-->   X(f) * exp(-i * 2*pi * f * tau)
```

The magnitude is completely unchanged, because `|exp(-i*theta)| = 1` for any real `theta`.
Only the phase moves, and it moves by:

```
delta_phi(f) = -2*pi * tau * f
```

**That is the ramp.** Read it as a function of `f`: it is a *straight line through the origin*
with slope `-2*pi*tau`. "Ramp" is just signal-processing jargon for "a quantity that grows
linearly," i.e. a straight line when you plot it.

```
  delta_phi
     ^
   0 +-------------------------> f
     |\
     | \
     |  \          slope = -2*pi*tau
     |   \
     |    \
```

The key consequence:

> **A pure time delay is invisible in magnitude but shows up in phase as a straight-line tilt
> that grows with frequency.**

Low frequencies barely shift; high frequencies shift a lot.

---

## 3. Why this destroys phase comparison here

Now put real numbers in. This experiment's grid: **2946 bins spanning 50–1000 Hz**, so about
0.32 Hz per bin.

Suppose two recordings start just **1 millisecond** apart (`tau = 0.001 s`). That is a tiny
offset — far below what you could control when triggering a camera and a speaker
independently. At the top of the band:

```
delta_phi(1000 Hz) = -2*pi * 0.001 * 1000 = -2*pi radians  = one FULL revolution
delta_phi( 500 Hz) = -2*pi * 0.001 *  500 = -pi   radians  = exactly out of phase
delta_phi(  50 Hz) = -2*pi * 0.001 *   50 = -0.31 radians  = barely moved
```

So **1 ms of jitter** — a rounding error in trigger timing — swings the phase at 1000 Hz
through a complete cycle, and flips the sign at 500 Hz. Two recordings of a *physically
identical* situation can have totally unrelated-looking phase.

It gets worse, because phase is an angle: it lives on a circle, mod `2*pi`. Once the ramp
exceeds one revolution, the measured value **wraps** back around:

```
true ramp:        0 ---- -pi ---- -2pi ---- -3pi ---- -4pi   (keeps descending)
measured (mod 2pi): 0 ---- -pi ----   0  ---- -pi  ----   0   (sawtooth, looks like noise)
```

Anything above ~1 ms of jitter therefore produces phase that *looks* random even though it is
perfectly deterministic.

Compare that nuisance to the thing we actually want to measure — how the box's response
changes when a cube moves a few centimetres. That is a subtle phase effect sitting underneath
a nuisance term that laps it several times over.

---

## 4. Why "sweep timing, not structure"

The chirp makes this sharper. In a linear sweep from `f_start` to `f_end` over duration `T`,
each frequency is emitted at one specific moment:

```
t(f) = T * (f - f_start) / (f_end - f_start)
```

For this experiment (`T = 3 s`, 50 -> 1000 Hz):

```
t(50 Hz)   = 0.00 s      <- emitted at the very start
t(500 Hz)  = 1.42 s      <- emitted in the middle
t(1000 Hz) = 3.00 s      <- emitted at the very end
```

So the phase at a given bin is set overwhelmingly by **when the sweep passed through that
frequency** — a property of the excitation *you designed*, not of the box. The structural
information is a small perturbation riding on top of a large, excitation-determined phase.

Hence the phrase: phase here is dominated by sweep timing, not by the structure.

---

## 5. The fix I tried, and why it should have worked

The logic is straightforward. *If* the dominant nuisance is a time delay, then by section 2 it
is **exactly** a straight line in phase — so estimate that line and subtract it off.

**Step 1 — isolate the phase difference between two samples.** The cross-spectrum multiplies
one spectrum by the conjugate of the other, which subtracts their phases:

```
P_ab(f) = conj(A(f)) * B(f)

angle(P_ab(f)) = phi_B(f) - phi_A(f)  ~=  -2*pi * (tau_B - tau_A) * f
```

Common structure cancels; what is left is dominated by the relative delay.

**Step 2 — undo the wrapping.** `np.unwrap` adds multiples of `2*pi` wherever the measured
angle jumps, converting the sawtooth of section 3 back into the straight line it should be.

**Step 3 — fit the line and remove it.** Weighted least squares for slope and intercept, then
multiply by the inverse of the fitted ramp:

```python
ph   = np.unwrap(np.angle(P))                        # undo mod-2pi wrapping
A    = np.vstack([bins, np.ones_like(bins)]).T       # design matrix: [f, 1]
sw   = np.sqrt(w)                                    # w = |log-SNR| salience weights
coef, *_ = np.linalg.lstsq(A * sw[:, None], ph * sw) # coef = [slope, intercept]
Pd   = P * np.exp(-1j * (coef[0]*bins + coef[1]))    # divide the ramp back out
```

The weights `w` matter: without them the ~98% of bins that are pure noise floor would drive
the fit.

**Step 4 — compare coherently.** With the ramp gone, sum the complex values instead of their
magnitudes, so agreeing phases reinforce and disagreeing phases cancel.

---

## 6. It failed — and the failure is the useful part

| approach | Spearman (more negative = better) |
|---|---|
| CPSD magnitude — **phase discarded** | **-0.526** |
| CPSD coherent — **after ramp removal** | **-0.136** |

Removing the ramp made things *dramatically worse*, not better. If a single time delay were
the whole story, this should have improved.

What that rules out: the corruption is **not a pure time shift**. Plausible culprits, none of
which is a straight line in phase:

- **Speaker group delay.** A real loudspeaker delays different frequencies by different
  amounts. That is a *curved* phase response, not a linear one.
- **Sweep-rate variation.** Playback clock and camera clock are not locked to each other.
- **Genuine nonstationarity.** A chirp is a transient; each mode is rung up and allowed to
  decay, so phase evolves during the record.

A straight line cannot represent any of these, so the fit absorbed real signal along with the
nuisance and left less behind than it removed.

---

## 7. What to do instead

**In this dataset: discard phase.** Use `np.abs()` and work in the magnitude domain. Resonance
and anti-resonance *frequencies* are properties of the structure and are invariant to every
timing problem above — which is exactly why the magnitude metrics win, with
`modal_peak_match` leading at -0.662.

**If you want phase to be usable, it is a protocol change, not a better fit.** Record a
**reference channel** simultaneously with the laser data — a microphone in the box, or a
loopback of the speaker's drive signal. Then:

1. Cross-correlate each recording's reference channel against the known chirp to *measure*
   `tau` directly, rather than inferring it from the data you are trying to analyse.
2. Align every recording to a common time origin before the FFT.
3. Better still, compute a true FRF as `H(f) = Response(f) / Reference(f)`. Dividing by a
   simultaneously-recorded reference cancels the excitation phase **exactly**, including the
   speaker's group delay and any sweep-rate wobble — the curved terms a linear fit cannot
   touch.

That last point connects to the deconvolution result in the notebook: dividing by the
*reference chirp file* barely helped (-0.520 -> -0.527) because that file is not what actually
reached the box. Dividing by the empirical cross-sample mean helped a lot (-> -0.652) because
it estimates the real, common excitation path. A recorded reference channel would give you the
same correction per-sample and in phase as well as magnitude.
