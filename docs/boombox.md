# Boombox on gastronorm

Adapting **The Boombox: Visual Reconstruction from Acoustic Vibrations** (arXiv 2105.08052)
to this capture. Living document: predictions are written *before* the run, results filled in after.

Run everything with `./scripts/boombox_ladder.sh`.

## What the paper does

4 contact mics -> a 128x128x4 mel-spectrogram (mics stacked as channels) -> fully-convolutional
encoder -> C-channel 1x1 embedding -> decoder whose every layer has two branches (a transposed
conv; and a conv then a transposed conv) concatenated -> RGB 128x128x3 + depth 128x128x1.
MSE on RGB, L1 on depth. Metrics: IoU on binarized masks, and a localization score = fraction of
samples whose predicted box center is within half the ground-truth box diagonal. 1575 sequences,
80/10/10, 3 seeds.

## What we changed, and why

**The lasers are the microphones, not the speakers.** Each of the 100 laser points measures wall
motion, so they are this capture's analogue of the 4 contact mics -- and all 100 come from *one*
camera capture at 2500Hz, so they are already frame-synchronized. The 8 speakers are 8 separate
*excitations* (one vibrates, stops, the next vibrates); that is the only axis with no shared time
base, which is why naive concatenation across speakers is not meaningful and why fusing them needs a
permutation-invariant pool rather than channel stacking.

**No spectrogram.** For an LTI box driven by a chirp, `Y(f) = H(f)X(f)` regardless of how the
excitation distributes energy in time, so time is a nuisance axis and a spectrogram adds no
information -- only the ability to reject the ~23% of the window with no excitation
(`docs/physics_and_architecture.md` sec 4). We feed the global rFFT the dataset already computes.

**No depth head.** The paper predicts RGB+depth because it has both. This dataset has one binary
occupancy mask and no depth channel anywhere, so an unsupervised depth head would be cargo cult.

**Frequency is what gets convolved.** Boombox downsamples 128x128 through seven stride-2 stages.
Our laser grid is 10x10, which supports about three -- a pooling layer, not an encoder. Frequency is
the long axis (1235 bins) and the one with real locality: resonance peaks and anti-resonance dips
are narrow-band. The encoder therefore convolves frequency with kernel height 1 (so lasers never mix
early), collapses it, and only then reshapes to the physical 10x10 grid to mix spatially. Mixing
lasers early would be wrong: adjacent lasers sit ~100px apart on the wall and sample global standing
waves, not local texture, so the grid is not translation-covariant the way an image is.

## Metrics added

`hard-iou` (IoU at threshold 0.5) and `localization` (the paper's score) now live in
`create_metrics`, so both models report them. Boxes are derived from the masks, not metadata: this
capture has no per-object bboxes, and `info['x_com'/'y_com']` are the `-1.0` sentinel for every
sample because the `downsampled_com` key is absent. `soft-iou` is *not* comparable to the paper's
published numbers; `hard-iou` is.

## Ladder

| # | question | variable | prediction |
|---|---|---|---|
| R0 | does it train at all? | 128 samples, 300ep | train soft-iou > 0.9, else the decoder is broken |
| R1 | does the CNN beat the transformer? | `--model` | **transformer wins or ties** |
| R2 | is boombox capacity-limited? | `--d-model` 512/1024/2048 | flat; 10x10 spatial extent is the bottleneck, not width |
| R3 | does the CNN want a higher LR? | `--lr` 1e-4/3e-4/1e-3 | 3e-4 best; CNNs usually want more than the transformer default |
| R4 | does the winning preprocessing transfer? | `log_magnitude` + empty-box + speaker | helps, same direction as the preprocessing ladder |

Decide on `eval/1-cube` and `eval/2-cubes`. Both are position-clean.

**Not decision metrics.** `eval/1-cube-speaker` and `eval/2-cubes-speaker` share *all* their
positions with train by design -- they answer "unseen speaker at a seen position", not
generalization to new positions. `eval/3-cubes` is 40 samples over 5 positions in 1 layout, so a
5-point swing is ~1 sigma; same for `count-acc-macro`, which averages over that 40-member class.

### Leakage audit (measured 2026-08-24)

| split | n | positions also in train | exact (pos,speaker) pairs in train |
|---|---|---|---|
| eval/1-cube | 112 | **0/14** | 0/112 |
| eval/2-cubes | 96 | **0/12** | 0/96 |
| eval/3-cubes | 40 | 0/5 | 0/40 |
| eval/red-cube | 40 | 0/5 | 0/40 |
| eval/1-cube-speaker | 63 | 42/42 (by design) | 0/63 |
| eval/2-cubes-speaker | 55 | 33/33 (by design) | 0/55 |

No recording appears on both sides of any split. The `*-speaker` splits are fully
position-leaked on purpose -- that is what "unseen speaker, seen position" means -- so they are
excluded from every generalization claim here. An older 12/12 leak in `eval/1-cube` has since
been repaired; it now measures 0/14.

**No split tests an unseen speaker identity.** All 8 speakers appear in train and every eval
split draws from all 8, so nothing currently measures transfer to a new speaker or device.

### How speakers are handled

R1b does **not** combine speakers. Each `(position, speaker)` recording is one independent
sample, exactly as the transformer treats it -- `fuse_speakers=False` is the default and
`--speakers-per-sample` does not exist yet. All 8 speakers are present, as 8 separate samples per
position, never merged. `BoomboxModel` has the `fuse_speakers` hook and the permutation-invariant
mean pool ready for when the grouped dataset is built; nothing calls it today.

## Results

### Final ladder results on `eval/2-cubes` (v2, 1000ep, all converged to +/-0.003)

| run | soft-iou | hard-iou | localization | com-dist | train-iou | gap |
|---|---|---|---|---|---|---|
| **R2 boombox d512** | **0.284** | **0.245** | 0.885 | 0.098 | 0.970 | 0.687 |
| R3 boombox lr3e-4 | 0.250 | 0.200 | **0.917** | 0.099 | 0.975 | 0.726 |
| R1 boombox d1024 | 0.249 | 0.188 | 0.833 | 0.113 | 0.971 | 0.721 |
| R3 boombox lr1e-3 | 0.235 | 0.183 | 0.906 | **0.093** | 0.977 | 0.742 |
| R2 boombox d2048 | 0.228 | 0.172 | 0.813 | 0.113 | 0.972 | 0.744 |
| R1 transformer | 0.163 | 0.116 | 0.813 | 0.097 | 0.360 | **0.196** |
| R4 boombox logmag+eb+spk | 0.152 | 0.100 | 0.490 | 0.141 | 0.969 | 0.817 |

## Log

**R0 passes.** Two checks, both on the gate question "can this architecture fit anything at all":

*Synthetic memorization* -- 16 fixed random inputs mapped to 16 distinct 4x4 blob targets,
400 AdamW steps at 3e-4:

```
step   0  loss 0.6917  soft-iou 0.025
step  80  loss 0.0212  soft-iou 0.447
step 160  loss 0.0010  soft-iou 0.963
step 399  loss 0.0001  soft-iou 0.995
```

So the encoder, the two-branch decoder, and the 3x4 -> 24x32 -> 21x30 resize all carry gradients
and the model has the capacity to place blobs at arbitrary positions. A failure here would have
meant the seed-grid projection or the interpolate was broken.

*Real data* -- 128 samples, `--lr 3e-4`, `--compile 0`, 300ep: train soft-iou climbs
0.020 -> **0.989**, hard-iou 0.980, localization 1.000, mse ~0. Cold-start soft-iou of ~0.02 is
just the mask density, i.e. the value for predicting nothing.

The eval numbers from this run (eval/1-cube soft-iou 0.096) carry **no signal** and must not be
compared against anything. R0 trains on 128 samples for 300 epochs *in order to* memorize them, so
its eval score is measuring a deliberately overfit model on held-out positions. R1 is the first
rung that produces a real generalization number.

This run also confirms `hard-iou` and `localization` register sane values end-to-end under
Composer, not just in the unit test.

Model is 27.7M parameters at `--d-model 1024`, against the transformer's much smaller default.
That size gap is itself a confound for R1 and is what R2 is for.

## Case study: why `bb-r2-boombox-d512-v2` misses a cube on sample 2593

Sample 002593 = position 325, speaker 1, 2 objects (purple + green), layout `purple--green-cube-grid4`,
in `eval/2-cubes` (a position-clean split).

**What the model actually does.** It is not predicting "one cube" in the sense of committing to a
1-object scene. It finds the lower-left cube well (pred blob at row 9.6, col 5.2 vs GT row 10.3,
col 4.7) and puts *some* mass at the upper-right cube's true location (rows 4-5, col 16-17) that
peaks around 0.2-0.5 -- under the 0.5 threshold, so it vanishes when binarized. Predicted mass is
6.13 vs 9.44 true: it under-paints by ~35%.

**The missed cube's signature IS in the spectrum.** Ranking all 70 speaker-1 one-cube samples by
log-magnitude cosine similarity to 2593 (after speaker-mean subtraction) and asking where each
one's cube physically sits:

```
Spearman(similarity, distance to MISSED cube): -0.450
Spearman(similarity, distance to FOUND  cube): -0.097
```

The strongest correlation is with the cube the model *missed*. The three most similar one-cube
samples (000345, 000209, 000361) all have their cube within 53-151px of the missed position. So
this is **not** a case of the second cube leaving no trace -- its trace is the single most
recoverable thing in the residual.

**It is also not "the two-cube spectrum looks like a one-cube spectrum."** That was the obvious
hypothesis and it is wrong, but only *after* the right preprocessing:

| feature | sim to 70 one-cube | sim to 285 two-cube | separation |
|---|---|---|---|
| raw log-magnitude | +0.9258 | +0.9239 | **-0.0019** |
| after speaker-mean subtraction | -0.0306 | +0.0593 | **+0.0899** |

On raw log-magnitude the classes are indistinguishable (separation -0.002, i.e. 2593 looks
*marginally more* like a one-cube sample). The speaker chain dominates everything. Subtracting the
speaker mean flips the sign and opens a real gap. Global 1-vs-2 effect size on speaker 1 is
d/std = 0.32 -- present, but small.

**Noise is not the explanation.** The same physical scene re-measured through 7 other speakers gives
cosine +0.80 to +0.88 against 2593. That is the repeatability ceiling, and it sits far above the
+0.09 class separation -- so the measurement is far more repeatable than the 1-vs-2 distinction is
large. The signal is stable; the *class difference* is what is small.

**The failure generalizes, and it is under-painting, not miscounting.** Across all 96 `eval/2-cubes`
samples (every one has exactly 2 GT blobs):

```
predicted blob count:  0 blobs: 7 | 1 blob: 32 | 2 blobs: 55 | 3 blobs: 2
predicted mass / true mass: mean 0.636, median 0.652
under-painting (ratio < 1): 94.8% of samples
```

The model under-paints on 95% of samples. On the 39 samples where it emits fewer than 2 blobs, the
weaker GT blob gets max activation of only 0.051 (median 0.002) against 0.687 at the stronger one --
and only 12.8% have even 0.2 of signal there. **2593 is the mild case, not the typical one:** it
retained visible sub-threshold mass at the missed cube, while the median such failure has the second
cube fully extinguished, not merely dimmed.

**Source of confusion.** Two compounding causes, in order of size:

1. *Systematic mass under-prediction.* 95% of samples under-paint, mean ratio 0.64. Under `ce-pixel`
   the background dominates the gradient (a 21x30 grid is ~9 true pixels against 621 background), so
   the loss is minimized by hedging. When total predicted mass is capped below truth, the model
   spends it on whichever cube is easier and starves the other -- so a *mass* deficit shows up as a
   *count* error.
2. *A small class margin against a large nuisance.* 1-vs-2 separation is d/std = 0.32 and needs
   speaker-mean subtraction to exist at all, while the speaker chain alone moves cosine similarity
   by ~0.9. The second cube's evidence is real but weak relative to what must be normalized away.

**What this predicts.** `--loss-fn ce-pixel-asym --loss-alpha > 0.5` (up-weights false negatives, so
under-painting is penalized) and `--subtract-speaker-mean` with `log_magnitude` should both help, and
should help *more* than added capacity. Note R2 varies `--d-model` only, which addresses neither --
consistent with capacity not being the bottleneck.

## Viz: why a real prediction renders as blank

Reported from the dashboard on 002593 (`bb-r2-boombox-d2048-v2`): the upper-right cube shows
nothing, but hovering reports nonzero values there.

**Confirmed, and it is a rendering choice rather than a colour-scale bug.** In
`viz/render.py:colorize`, opacity is the value itself:

```python
t = v ** GAMMA          # GAMMA = 0.85
alpha = t               # a cell's opacity IS its own value
```

For 002593 under d2048:

| region | peak pred | alpha | over backdrop | RGB delta vs backdrop |
|---|---|---|---|---|
| lower-left (found) | 0.996 | 0.997 | 254/255 | 143.2 |
| upper-right (missed) | 0.163 | 0.214 | 46/255 | **13.6** |

A delta of ~13 on a light backdrop is around the threshold of noticing, and it sits next to a
neighbour at 143 -- so the eye adapts to the strong blob and the faint one reads as empty. The
value is on screen; it is just below perceptual threshold.

**The `rel` toggle does not fix it.** `domain_of` rescales by the mask's own min/max, but the max
here is already 0.996, so the relative domain is ~[0,1] -- identical to the fixed one. Measured:
alpha 0.214 (fixed) vs 0.215 (relative). The relative view is a no-op in exactly the case that
motivates it: one confident blob pinning the top of the range while the other is faint.

**Prevalence: 73 of 96 (76%) `eval/2-cubes` samples have one cube rendering below 30% opacity.**
This is not a one-sample quirk -- for a model that under-paints on 95% of samples, alpha-encodes-value
systematically hides the exact evidence needed to diagnose it.

**Options, cheapest first.**
1. Floor the alpha for any cell above a small value, e.g. `alpha = 0.35 + 0.65*t` where `t > 0.02`,
   so weak-but-real cells stay visible while true zeros stay clear.
2. Decouple opacity from value: keep alpha constant and let the colour ramp alone carry magnitude.
   Loses the "silhouette floats free" property the docstring wants.
3. Make `rel` use a percentile domain (e.g. 2nd-98th) instead of min/max, so a single saturated
   cell stops defining the scale.

**Fixed.** Three compounding causes, applied in order:

| change | 0.20 prediction renders at |
|---|---|
| original (`alpha = t`, `GAMMA = 0.85`, ramp starts `#ffffff`) | delta **13.6** (invisible) |
| + `ALPHA_FLOOR = 0.35` | delta 50.8 (marginal) |
| + `GAMMA = 0.5` | delta 99.6 (readable) |
| + trim the 2 palest ramp stops | delta **121.0** |

The floor alone was not enough, and the reason is worth recording: **alpha was never the
binding constraint.** At `GAMMA = 0.85` a 0.24 prediction landed on `#9ec5f4`-ish, within ~90
RGB units of the `[238,238,235]` backdrop *at full opacity* -- no floor value can rescue a
colour that close to the background. Lowering gamma moves faint values onto saturated colour,
and trimming the near-white stops off the bottom of the ramp means even the ramp's floor has
real colour. `SEQ_HEX` and `TRUE_SEQ_HEX` now start at `#cde2fb` / `#c9e8d6` instead of
`#ffffff`.

Cost: background cells rendering above the visibility threshold go from a median 6/630 to
16/630, and those are mostly the genuine halo around true blobs. True zeros still render at
alpha exactly 0, so an empty mask is still empty.

## Case study 2: sample 002497 -- a different failure from 2593

002497 = position 313, speaker 1, 2 objects, `purple--green-cube-grid4`, in `eval/2-cubes`.

**Harder failure than 2593.** The missed cube peaks at **0.0001** -- not sub-threshold, but
absent. (2593's missed cube peaked at 0.16, faint but present.) Predicted mass 4.99 vs 10.45
true, a ratio of 0.48.

**The missed cube's signature is genuinely weak here, unlike 2593.** Ranking all 70 speaker-1
one-cube samples by spectral similarity and correlating against distance to each object:

| sample | to MISSED cube | to FOUND cube |
|---|---|---|
| 2593 | **-0.450** | -0.097 |
| 2497 | -0.269 | **-0.575** |

These are opposite. On 2593 the missed cube was the *most* recoverable thing in the residual --
the evidence was there and the model failed to use it. On 2497 the found cube dominates the
spectral neighbourhood (-0.575) and the missed cube is much weaker (-0.269). Nearest one-cube
neighbours agree: those near the found cube correlate +0.062/+0.044, those near the missed cube
only +0.025/+0.031.

**Proximity is a plausible but UNPROVEN cause.** 2497's cube separation is 7.4 grid cells, the
second-tightest scene in the split (range 7.1-14.6).

The correct unit of analysis is the **position, not the sample**: the 8 speakers at a position are
repeat measurements of one scene, so `eval/2-cubes` is 12 independent scenes, not 96. A per-sample
binning looks like n=24 per quartile but the closest quartile is really ~3 distinct positions.

Per position (n=12):

| position | separation | mean weaker peak | 2-blob rate |
|---|---|---|---|
| 345 | 7.1 | 0.151 | 25% |
| 313 (**2497**) | 7.4 | 0.000 | 38% |
| 330 | 8.9 | 0.006 | 0% |
| 304 | 10.2 | 0.147 | 62% |
| 314 | 10.3 | 0.775 | 100% |
| 310 | 10.3 | 0.467 | 100% |
| 291 | 11.2 | 0.299 | 50% |
| 341 | 12.2 | 0.605 | 75% |
| 303 | 12.3 | **0.035** | **12%** |
| 293 | 13.3 | 0.366 | 75% |
| 325 (2593) | 14.1 | 0.243 | 50% |
| 336 | 14.6 | 0.808 | 100% |

`Spearman(separation, weaker peak) = +0.538` (n=12, **p~0.07 -- not significant**). The three
tightest scenes are the three worst, which is suggestive. But **position 303 at separation 12.3 is
mid-range and fails as badly as the tightest pairs** (peak 0.035, 12%), which a pure proximity
mechanism does not explain.

Separation is fully confounded with position here -- grid4 offers no scenes that vary separation at
a matched position -- so "the cubes are close" cannot be separated from "this position is hard".
Treat proximity as a hypothesis consistent with the physics
(`docs/physics_and_architecture.md` sec 2: nearby perturbations couple through the same local modes,
and the cross terms are first-class) but not as an established cause.

**Two dead ends, ruled out.** (a) *Label quality*: SAM segmentation scores run on the overhead
photo, never on the vibration input, so they cannot cause a model failure --
`Spearman(SAM score, model peak) = -0.116`, i.e. nothing. (b) *Colour bias*: purple objects peak
at 0.509 vs green 0.624 on average, a modest gap with no mechanism behind it. Neither explains
2497.

**Cross-speaker reproducibility is low for both, and this is the deeper problem.** With the
speaker mean removed, re-measuring the same physical scene through the other 7 speakers gives
only +0.194 (2497) and +0.102 (2593), against a random 2-cube-pair baseline of +0.054. So after
the speaker nuisance is divided out, what remains that is specific to a scene is barely above
chance. That is the ceiling every model here is working under.

## Is 2497's FFT magnitude itself an outlier?

Looking at the signal directly (raw `|Z|`, not correlations against other samples), against the
375 speaker-1 samples.

**Globally: no.** Every summary statistic is within 1 sd of the pool.

| stat | 2497 z | 2593 z |
|---|---|---|
| mean \|Z\| | -0.85 | +0.29 |
| median | +0.31 | -1.00 |
| std | -0.47 | +0.14 |
| max | +0.19 | -0.08 |
| p99 | -0.36 | +0.46 |
| total energy | -0.39 | +0.08 |

**No dead or hot lasers.** Per-laser energy z-scored against the pool: 0 of 100 lasers exceed
|z|>3 in either sample, and 0 lasers are dead (<1% of that laser's pool median). For contrast,
real capture failures DO exist in this dataset -- 002881 and 002849 have **100/100** lasers at
|z|>3 with total energy z=+12.6 -- so the check is sensitive; 2497 simply is not one of them.
2497 ranks 303/375 by number of outlier lasers.

**The one real anomaly is spectral SHAPE, not level.** 2497's energy is redistributed across
the band:

| band | 2497 z | 2593 z |
|---|---|---|
| 50-200 Hz | -0.43 | +0.51 |
| 200-400 Hz | **-1.20** | -0.58 |
| 400-600 Hz | -0.97 | +0.87 |
| 600-800 Hz | +0.88 | -0.85 |
| 800-1000 Hz | **+1.12** | -1.05 |

Summarising as tilt = `log(E[600-1000] / E[200-600])`, **2497 sits at the 96th percentile of the
speaker-1 pool (z=+1.86)**: depleted in the mid band, elevated at the top. Its 6 most anomalous
individual bins are all 837-985 Hz (z up to +4.7). 2593 is the mirror image (17th percentile,
anomalous bins all at 64-118 Hz).

**But tilt only weakly predicts failure, and 2497 is NOT a tilt outlier within its own split.**
Across `eval/2-cubes`, `Spearman(tilt, weaker cube peak) = -0.244`:

| tilt quartile | weaker peak | 2-blob rate |
|---|---|---|
| least tilted | 0.461 | 67% |
| q2 | 0.320 | 58% |
| q3 | 0.313 | 62% |
| most tilted | 0.206 | 42% |

A real monotone trend, but modest. And 2497 is only the **23rd percentile of tilt within
`eval/2-cubes`** -- the eval split is systematically more tilted than the pool 2497 looked extreme
against. So tilt does not single 2497 out among its actual peers.

**Layout-level tilt shift is small.** grid4 (the eval/2-cubes source) sits +0.052 above the
always-train layouts, only +0.20 sd -- not a distribution shift large enough to explain the
failure. Note grid1, which is always in train, is the *least* tilted layout (-2.246) while grid4
is among the most (-2.111); that is a mild train/eval mismatch worth remembering but not a
smoking gun.

**Conclusion: 2497's input is not defective.** No bad lasers, normal energy, normal dynamic
range. It has an unusual spectral tilt relative to the broad pool, but a perfectly ordinary one
relative to the split it is evaluated in. Whatever makes it hard is not a corrupted or
out-of-range measurement.
