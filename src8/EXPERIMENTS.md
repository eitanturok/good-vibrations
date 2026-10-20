# src8: ModalFormer, an interpretable laser x frequency transformer

Goal: match or beat `s7-gpc-tukey-logpower-1k` on **IoU and miss at threshold 0.08**, using an interpretable architecture that
generalizes to unseen objects, speakers and lasers (the box is always known). It should pay attention to object contour and
rotation, and its output head should be swappable (mask now; rgb or depth later).

Scoring (`src8/score.py`, any run's saved eval outputs): pred > 0.08 vs true > 0.5 →
- **iou@0.08**: hard IoU per sample.
- **miss@0.08**: no pixel over 0.08 lands on the object.

Each split is a mean over its samples. **ALL** = mean over the 54 non-control eval splits (every split counts once, as in EvalSummary). Also id / ood / per box.
Analysis notebook: `notebooks/105_modalformer.ipynb`.

## 0 Target

| run | epoch | ALL iou@.08 | ALL miss@.08 | id iou | ood iou | ood miss |
|---|---|---|---|---|---|---|
| s7-gpc-tukey-logpower-1k (target; crashed at ep751) | 750 | **0.299** | **0.275** | 0.581 | 0.219 | 0.341 |
| s7-gpc-tukey-logpower-1k | 500 (only ckpt) | 0.300 | 0.272 | 0.580 | 0.220 | — |
| s7-gpc-tukey-base-1k (magnitude) | 750 | 0.328 | 0.245 | 0.627 | 0.243 | 0.306 |

## 1 Literature: what matches our data

Our problem: about 11k samples. Each is 100 sensors on a fixed 10x10 grid on one side wall, times 1170 frequency bins (0.77 Hz),
times 2 axes. The output is a 64x64 floor mask. It's an inverse problem of the "response" type: an active chirp excites a fixed
box, and the object perturbs the box's modes.

| idea | paper | what we take |
|---|---|---|
| sparse sensors → dense field through a latent + per-output queries | [Perceiver IO](https://arxiv.org/abs/2107.14795) (ICLR'22); [Senseiver](https://neurips.cc/virtual/2022/56953) (NeurIPS'22 / Nat. Mach. Intell.'23) | pixel-query decoder: the output is a set of floor-coordinate queries. That makes it task-agnostic (mask / rgb / depth = the head's channels) and resolution-flexible (Fourier coordinates) |
| boundary measurements → interior image with attention | [Transformer Meets Boundary Value Inverse Problems](https://arxiv.org/abs/2209.14977) (EIT, ICLR'23) | closest physics analogue (sensors on the boundary, interior unknown); attention as a learned non-local kernel |
| per-axis attention on spectrograms | [SepTr](https://arxiv.org/abs/2203.09581) (Interspeech'22); axial attention | factorized attention: over lasers within a frequency band (a *mode shape*), and over bands within a laser |
| sensor tokens with position embeddings, robust to layout | [SPAR](https://arxiv.org/abs/2505.16936) (2025) | per-laser learned + Fourier grid embedding; laser dropout for unseen laser placements |
| masks from queries; contour losses | [Mask2Former](https://arxiv.org/abs/2112.01527); [SDF regression for shape-aware segmentation](https://arxiv.org/abs/1912.03849) (AAAI'20) | SDF channel + outline channel as contour supervision |
| rotation of contours | [RotaTouille](https://arxiv.org/abs/2508.16359) (2025) | rotation needs an explicit signal. We supervise mask second moments (orientation) directly |
| classic: modal analysis / damage localization | Rayleigh perturbation; mode-shape curvature (Pandey 1991); FDD (SVD over sensors per frequency); see `lit_review.md` | the physics below |

Earlier attempts in this repo:
- src VibrationTransformer port (cls pooling + MLP decoder): plastic iou@.08 **0.304 vs conv 0.411**.
- Slots and polar heads: below the mask head.

Lesson: a transformer needs a local frequency stem and a spatial decoder, not one cls vector.

## 2 Diagnostics before designing (what makes the current model tick)

**Probe** (`src8/probe.py`, baseline ep500, all 4560 eval samples, IoU@0.08):
- none 0.300
- freq roll 1% (9 Hz) 0.056; freq flip / roll 10% / block shuffle ≈ 0.01
- shuffle within 9 bins 0.168; smooth over 33 bins 0.157
- level only 0.003
- remove the laser mean 0.079
- laser shuffle 0.019; grid roll 1 column 0.117; swap x/y 0.011

So the conv model is NOT frequency-agnostic, even though its convs share weights over frequency and it mean-pools frequency
(strided convs alias, and that leaks absolute position). The signal is the *precise* peak positions: a few Hz, which is
exactly what modal theory predicts (object → Δω_k of a few Hz). It also reads laser identity, and the common-mode spectrum.

**Distance saturates**: per-band spectral distance vs object-position gap (nb101's `S(Δpos)`) rises over 0–0.15 of the box,
then is flat. That holds in every band, from 100–190 Hz to 640–1000 Hz. Low bands keep rising slightly longer.

**No locality**: which lasers change most does not track where the object is (|ρ| ≤ 0.07). Modes are global, so a fixed
"nearby laser" prior would be wrong. Any floor↔wall prior must be learnable and allowed to switch off.

**Rotation is in the signal; the model drops it** (hardest case: rotated soap dispenser in cardboard). Orientation = principal axis of
the true mask (doubled angle, chance error 45°).
- Baseline eval: cardboard soap-rotated orient err **36.5°**, upright 11.4°; gastro/plastic sideways 81°/71°. It paints a soap-shaped blob at the training orientation.
- Linear ridge probe (spectrum → cos2θ, sin2θ; 200 PCs; GroupKFold over positions; scratchpad `rot_probe.py`):
  - cardboard rotated **16.9°**, rotated-2 **20.8°**, sideways 51°.
  - Position alone (kNN on com) gives 27–39°, and the constant mean gives 33–35°.
  - Plastic sideways stays at 84° (no rotated training data, so not learnable there).

So for cardboard rotation the bottleneck is the model, not the signal. That motivates T5's moment loss, and if needed an explicit orientation head.

**Modal analysis intuition.**

  X_l(f) ≈ S(f) Σ_k φ_k(x_l) φ_k(x_s) / (ω_k² − ω² + 2iζ_k ω_k ω),    δω_k/ω_k ≈ −½ ∫_obj ρ φ_k(x)² dx / M_k

1. Each mode's shift measures the footprint projected on φ_k². Low modes give position and area; high (weak) modes give orientation and contour. That is why shape and rotation are hard (nb97: predicted shapes ≈ disk) and why cardboard is hard: damping widens the peaks, so shifts blur (nb99).
2. The laser pattern at a frequency identifies *which* mode; the frequency says how far it moved. So read lasers per band (mode shape), and read bands per laser.
3. The speaker enters as S(f)·φ_k(x_s). Mode frequencies, widths and laser ratios are speaker-invariant. In the log domain, subtracting the mean over lasers cancels S(f) exactly.
4. φ_k² oscillates on the mode's wavelength, so the raw distance saturates beyond about a wavelength. The combination of modes is still unique (like a Fourier positional code), so keep the bands separate.

## 3 Gameplan: 5 theories

| # | theory | mechanism in src8 | flag(s) | prediction |
|---|---|---|---|---|
| T1 | **The one-vector bottleneck under-paints**: the conv model squeezes everything into one 1024-d vector, then a deconv decoder | pixel queries (16x16 floor grid, Fourier coords + box) cross-attend to 100 laser tokens, then 2 TwoBranchUp + a 3-conv head | `--encoder modal` | fewer misses on ood; output head swappable (rgb/depth = channels) |
| T2 | **Modal tokenization**: keep (laser, band) tokens with explicit Hz and laser identity, and interleave laser-attention (mode shapes) with band-attention | stem = FreqEncoder conv1-3 + log f channel → 19 band tokens of 64 bins per laser; axial blocks `--modal-order` (LFLF / FFFF / LLLL); attention-pooling over bands | `--modal-order` | LFLF ≥ FFFF (freq-first, ≈ today's conv) and ≥ LLLL; band-pool weights = which frequencies matter, cross-attention = which lasers each floor cell reads |
| T3 | **Floor ↔ wall alignment prior** (user idea) | per box, a learnable affine map of the laser grid onto the floor; head h adds −α_h‖q − p(r,c)‖², α_h learned (init 0.1…30) | `--geo 1/0` | small gain; the learned α tells us whether locality exists at all (the diagnostic says it barely does) |
| T4 | **Invariances for unseen speakers / lasers** | (a) laser-centering in log power (cancels S(f)); (b) random smooth spectral tilt augmentation; (c) laser dropout | `--laser-center`, `--color-aug`, `--laser-drop` | unseen-speaker and shifted-laser splits improve; id may drop slightly |
| T5 | **Contour + rotation supervision** | extra channels: outline (bce+dice, as src7 edge, which gave +0.017) and SDF (tanh, ±6 px); moment loss on single objects (log area, centroid, covariance = size + orientation); new metric `orient_err` | `--edge-weight`, `--sdf-weight`, `--moment-weight` | better boundary IoU and orient_err, fewer misses (the area term fights under-painting) |
| (T6) | Distance saturates, so the signal is a local kernel: retrieval / prototype attention over a memory of training embeddings | — | — | candidate if time permits |
| (T7) | Multi-speaker fusion: the 4 speakers at one position are 4 views (different φ_k(x_s) weights, independent noise) | set attention over speaker captures | — | candidate: probably the biggest untested lever |

Preprocessing candidates considered:
- empty-box subtraction: dropped. A Spearman proxy liked it, but trained runs and a ridge probe ranked it below plain log (2026-09-27).
- absolute phase: dropped (hurts, overfits; phase_src7.sh).
- IR gating: dropped (nb100).
- Farina harmonic IRs as extra channels (rattle/contact): untested.
- x/y polarization (mode-shape gradient direction): untested.

## 4 Runs

**Epochs: 500, not 1000.** Rankings at ep500 match ep1000 in every past sweep (Spearman of ALL iou@.08: gpc 1.0, plastic 0.98, gastro 0.8,
cardboard 0.8; 25 runs, `runs/ep500_vs_1k.csv`). The gain from ep500 to ep1000 is −0.005…+0.013, and cardboard runs drop. The only
reshuffles are within noise (the gastro 0.343–0.352 cluster). The target's own ep500 score is 0.300 vs 0.299 at ep750.

Headline = **CORE** (vase / cube / soap / candle × 3 boxes, 12 splits) + **HARD** (cardboard rotated soap, with orientation error); ALL secondary.

### v1 (scripts/modal_src8.sh, retired): the pixel-query ModalFormer fit a constant mask
- s8-modal-1k was stopped at ep~110: eval iou@.08 0.074 vs the target's 0.281 at ep100. Its **train** mask bce was 0.062 vs 0.017 at the same step, so it couldn't fit.
- `src8/fitdiag.py` (600 steps on the cached 4560 eval tensors, mask loss only): train bce stuck at 0.075 at lr 1e-4, 5e-4 and 1e-3, held-out iou 0.000. Without axial blocks or without geo: the same.
- Activation trace: across-sample std / total std is 0.56 after the stem, 0.05 after band pooling, 0.01 in the decoder. Gradient into the stem: 1e-4.
- **Cause 1, band attention-pooling = a weighted MEAN over 19 bands.** It washes the per-band content out (≈1/√19), while the laser embedding, identical in every band, survives intact. **Fix:** gated flatten: each band token × its softmax gate × P, concatenated → Linear. The gates are still an interpretable per-laser band weight.
- **Cause 2, the embedding was a mean over lasers**, which loses laser identity (the probe: laser shuffle → 0.02). **Fix:** flatten → Linear.
- Per-box input normalization ((x − μ_box)/σ_box per laser, bin, axis; running stats) is kept: the object is a small perturbation of the box's own spectrum. It alone did not unblock the fit (0.075 → 0.075), but it helps a little once pooling is fixed (0.032 → 0.027).

fitdiag after the fixes (train bce / held-out iou@.08, 600 steps; single seed, so differences under ~0.04 are noise):

| config | train bce | held-out iou | time |
|---|---|---|---|
| conv (lr 1e-4) | 0.0171 | 0.390 | 61 s |
| conv lr 3e-4 / 5e-4 | 0.0118 / 0.0107 | 0.376 / 0.400 | 61 s |
| query ModalFormer, LFLF, 2 decoder layers, lr 1e-4 / 5e-4 | 0.0316 / 0.0269 | 0.248 / 0.251 | 178 s |
| ... − box norm | 0.0319 | 0.255 | 172 s |
| ... no axial blocks | 0.0275 | 0.242 | 71 s |
| ... d 256 | 0.0257 | 0.286 | 285 s |
| ... **4 decoder layers** | 0.0210 | 0.326 | 190 s |
| ... 6 decoder layers | 0.0316 | 0.240 | 202 s |
| hybrid (ModalFormer front end → conv decoder), LFLF, lr 5e-4 | 0.0350 | 0.289 | 165 s |
| hybrid, no axial, lr 5e-4 | 0.0235 | 0.374 | 60 s |
| **hybrid, no axial, lr 1e-4** | 0.0187 | **0.442** | 60 s |

The hybrid generalizes better than conv (higher train loss, higher held-out iou) at the same speed. That makes it the lead candidate.

### v2 (scripts/modal_src8_v2.sh, 500 ep, tmux train:src8)

| run | encoder → decoder | change | question |
|---|---|---|---|
| s8-hybrid-500ep | ModalFormer front end (no axial) → conv | — | lead candidate |
| s8-conv-logpower-500ep | conv → conv | target recipe | 500-ep control |
| s8-hybrid-contour-500ep | hybrid | + edge channel + moment loss | T5, rotated soap |
| s8-query-500ep | ModalFormer LFLF → 4-layer pixel queries + geo | fully attentional | T1/T3, interpretable |
| s8-hybrid-LFLF-500ep | hybrid + LFLF axial | mode-shape attention | T2 |
| s8-hybrid-inv-500ep | hybrid | + laser drop 0.2 + spectral tilt 0.3 | T4 |
| s8-hybrid-center-500ep | hybrid | + laser-centering | T4 |
| s8-conv-logpower-z-500ep | conv | --norm z | is log power's gap its loudness offset? |
| s8-conv-mag-500ep | conv | magnitude | input control |

### Signal probe: preprocessing without training (`src8/signal_probe.py`, `runs/signal_probe_src8*.csv`)
Validated proxy: held-out-position kernel-ridge CoM skill. It ranked past trained runs correctly; label-free SNR did not.

| arm | com_skill card / plas / gastro | unseen-speaker skill | rotated-soap orient | speaker spread dB (card / gastro) |
|---|---|---|---|---|
| log power | 0.708 / 0.781 / 0.732 | 0.02–0.04 | 17.3° | 1.60 / 1.22 |
| magnitude | 0.694 / 0.761 / 0.722 | 0.03–0.08 | 17.2° | — |
| laser-centered | 0.702 / 0.784 / 0.736 | 0.02–0.03 | 17.1° | **1.34 / 0.82** |
| − empty box (box + speaker) | 0.705 / 0.783 / 0.732 | ~0 | 17.8° | 1.60 / 1.22 |
| deconv (/ chirp) | 0.707 / 0.781 / 0.732 | 0.02–0.04 | 17.3° | 1.60 / 1.22 |
| Farina 2nd + 3rd harmonics only | 0.388 / 0.683 / 0.601 | **0.05–0.09** | 26.1° | 2.9 / 2.7 |
| linear + harmonics concatenated | 0.607 / 0.757 / 0.708 | 0.02–0.05 | 21.6° | — |

- Centering, empty-box subtraction and deconvolution do **not** change decodability (±0.006, all boxes). Centering does cut the speaker spread 16–33%, but that doesn't make transfer better.
- Unseen-speaker transfer is ≈ 0 for a linear probe with *any* preprocessing. The fingerprints are speaker-specific.
- The harmonics carry independent position information (skill 0.39–0.68 on their own) and transfer best across speakers. They're a candidate extra input channel for a trained model.
- **Multi-speaker fusion barely helps.** Same positions and count (cardboard 1277, plastic 1103 positions), com_skill: speaker 1 alone 0.572 / 0.567, mean of 4 speakers 0.576 / 0.571, concatenation of 4 0.579 / 0.578. That's +0.004–0.011, so T7 is downgraded: averaging speakers isn't a big SNR win, consistent with the noise being speaker-geometry drift rather than white noise (nb101). Gastro can't be tested (its positions don't group all 4 speakers under one id).
- Farina harmonics are now an input option: `--window farina` = the tukey spectrum + the 2nd-harmonic IR spectrum (C 2 → 4). The harmonic channels are rescaled by the chirp's median |C| so they clear the log floor (before: 62–86% of bins under 1e-6; after: 0.5%; about 31 dB below the linear response).

## 5 What worked / what didn't

### Results (500 ep unless noted; target = s7-gpc-tukey-logpower-1k @ep500)

| run | CORE iou | CORE miss | HARD iou | HARD miss | HARD orient | ALL iou | ALL miss |
|---|---|---|---|---|---|---|---|
| target (s7 conv, log power) | 0.580 | 0.043 | 0.420 | 0.024 | 36.3° | 0.300 | 0.272 |
| s8-conv-logpower (target recipe, 500-ep schedule) | 0.577 | 0.046 | 0.408 | 0.032 | 36.4° | 0.299 | 0.266 |
| **s8-hybrid-base** | **0.643** | **0.029** | **0.518** | **0.016** | **30.2°** | **0.336** | **0.252** |
| **s8-best-mag** (spkgain on magnitude input instead of log power): **new best** | **0.738** | **0.005** | **0.597** | **0.000** | **17.1°** | **0.399** | **0.163** |
| **s8-best-spkgain** (jitter + physics-matched speaker gain): previous best | **0.722** | **0.006** | 0.596 | **0.000** | 18.3° | **0.388** | **0.168** |
| s8-best-jitter (contour + laser jitter ±0.5 cell) | 0.705 | 0.009 | **0.603** | **0.000** | **16.1°** | 0.367 | 0.220 |
| s8-best-tversky (best + bce-Tversky, FN 0.7 / FP 0.3) | **0.698** | 0.012 | 0.579 | 0.016 | 18.7° | **0.359** | **0.197** |
| **s8-posesum-contour** (posesum + outline channel + moment loss) | **0.689** | **0.011** | **0.579** | 0.024 | **17.6°** | **0.355** | **0.202** |
| s8-hybrid-posesum (pose branch, residual merge, mask soft-argmax centre) | 0.662 | 0.025 | 0.550 | **0.008** | 21.3° | 0.342 | 0.255 |
| s8-hybrid-pose (STN pose branch, soft-OR merge, com-head centres) | 0.570 | **0.012** | 0.454 | **0.000** | 29.0° | 0.294 | **0.199** |

s8-hybrid-base trajectory (CORE / HARD iou, HARD orient):
- ep100: .595 / .524 / 28.5°
- ep200: .632 / .532 / 28.1°
- ep300: .643 / .520 / 29.9°
- ep500: .643 / .518 / 30.2°

Rotation is best early and drifts back toward the training orientation as training continues.

The 500-ep conv control reproduces the target (CORE .577 vs .580), so the hybrid's gain is the architecture, not the schedule.

**Hybrid vs the 500-ep conv control, CORE:**
- It wins in **all 12 object × box cells**: cardboard cube +.144, soap +.100, vase +.097, candle +.049; gastro +.016–.087; plastic +.028–.075.
- It wins for **every speaker**.
- The gain grows with difficulty: largest in the damped cardboard box, smallest where conv was already good.
- CoM error is lower in every cell (cardboard cube .081 → .048), and the cube is under-painted less (area .60 → .77).
- Its only consistent loss is the unseen speaker 4 in plastic (−.053).

### Magnitude input on the best model (s8-best-mag): the new best
Hypothesis: log power boosts the valleys between resonance peaks, and those are mostly the speaker's own response S(f). s6 (conv, magnitude) held the unseen speaker 4 in plastic at its seen level (.668); s8 on log power fell to .395.
- vs s8-best-spkgain @ep500:
  - CORE .738 / .005 vs .722 / .006, **better in all 12 object × box cells** (+.002 to +.044; plastic cube +.044);
  - HARD .597 / 0 vs .596 / 0, orientation 17.1° vs 18.3°;
  - ALL .399 / .163 vs .388 / .168; boundary IoU CORE .514 vs .491.
- Categories (split mean): **unseen speaker .355 vs .279**; unseen laser .215 vs .203; multiple .348 vs .329; broken .507 vs .495; unseen object .533 vs .526. Lid .077 vs .093 and cylinder .120 vs .125 lose slightly. Both were already near zero.
- Unseen speaker 4 by box: **plastic .568 vs .395** (+.173; s6 .668), cardboard .353 vs .317, gastro .143 vs .125. Gastro speaker 4 is still the worst group: it misses 26% and paints 2.9× the area.
- Two-object scenes: both objects found in .36 vs .29, union iou .345 vs .323.
- So the hypothesis holds. Magnitude keeps the peaks dominant over the valleys where the speaker lives. The later arms (slots, nopose, poseonly) are rebased on it.

### What each half of the residual merge does (s8-best-spkgain, eval-time ablation, no retraining)
The output is `direct + warped canonical`. Switching each half off, each at its own best threshold:

| output | CORE iou / miss | HARD iou / miss | HARD orient | ALL iou |
|---|---|---|---|---|
| sum (as trained, thr .08) | .722 / .006 | .596 / 0 | 18.4° | .406 |
| direct only (thr .005) | .709 / .006 | .572 / 0 | **33.6°** | .410 |
| pose only (thr .98) | .712 / .007 | .573 / 0 | **19.0°** | .373 |

- Training split the work on its own. The direct decoder became a locator and background suppressor: logits −14.8 off the object and −3.8 on it, so its mask alone sits below 0.5. The pose branch draws the shape: +5.7 on the object, ≈0 off it.
- **Rotation lives entirely in the pose half.** The direct half alone is at 33.6°, the no-pose hybrid's level (30.2°).
- Pose-only loses ~.01 CORE, ~.02 HARD and .03 ALL. ALL holds the multi-object / broken / lid scenes that one rigid slot can't draw.
- Pose-only still takes its centre from the direct map's soft-argmax, so the direct map's real job is the centre heatmap.

### Contour + moment supervision on top (s8-posesum-contour): the current best
- posesum + `--edge-weight 1` (outline channel) + `--moment-weight 1` (L1 on log area, centroid, 100 × covariance of the soft mask, single objects).
- @ep500 vs posesum:
  - CORE .689 / .011 vs .662 / .025
  - HARD .579 / .024 vs .550 / .008
  - orientation HARD 17.6° vs 21.3°, CORE 4.5° vs 9.3°
  - ALL .355 / .202 vs .342 / .255
- Decomposition: shape after position (+ size) alignment improves (CORE .836 / .895, HARD .734 / .769), and CoM error is unchanged (.023).
- At ep100 it looked worse (HARD .430, CoM error .051 vs .033): the shape losses come first and placement catches up later. A reminder not to judge new loss terms at ep100.
- A variant without the centroid term (`--moment-terms as`) is implemented but not needed so far.
- **+ Tversky loss** (s8-best-tversky): a small, consistent gain.
  - CORE .698 vs .689, HARD miss .016 vs .024, ALL .359 / .197 vs .355 / .202; orientation ≈ (18.7° vs 17.6°).
  - At ep300, HARD reached .602 with 0 misses.
  - About the size of run-to-run noise, so it's a mild positive and not yet the new base.
- By category vs the target (iou@.08):
  - Gains: id +.110 (.690), rotated +.124 (.532), unseen rotated object +.144 (.367), broken +.284, multiple +.061, unseen object +.030; misses fall in all of these.
  - Physics-limited, barely moved: cylinder +.018 (.117), lid +.019.
  - Unseen laser: flat (.113 vs .110).
  - **Unseen speaker: worse (.199 vs .260; gastro speaker 4 .036).** The richer per-band, per-laser features fit the training speakers' fingerprints more tightly. Queued: the physics-matched speaker-gain augmentation and laser jitter.

### Speaker gain on top of jitter (s8-best-spkgain): the best, and every target beaten
- jitter + `--color-aug 0.35 --color-ell 20`: a random log-gain g(f) per sample, shared by all lasers and both axes, a smooth random curve with knots every 20 Hz and rms 0.35. This is the measured structure of the difference between two speakers' S(f) (the speaker multiplies every laser's response by the same S(f), so in log power it adds the same curve everywhere).
- @ep500 vs jitter:
  - CORE .722 / .006 vs .705 / .009
  - HARD .596 / 0 misses vs .603 / 0 (≈; orientation 18.3° vs 16.1°)
  - ALL .388 / .168 vs .367 / .220
- By category (iou@.08; target / jitter / spkgain):

  | category | target | jitter | spkgain |
  |---|---|---|---|
  | **unseen speaker** | .260 | .228 | **.275** |
  | unseen laser | .110 | .170 | **.201** |
  | multiple | .267 | .251 | **.323** |
  | lid | .032 | .054 | **.096** |
  | unseen rotated object | .223 | .398 | **.436** |
  | unseen object | .508 | .565 | .574 |
  | id | .580 | .707 | .723 |
  | rotated | .409 | .564 | .551 |
  | broken | .190 | .577 | .495 |

- Unseen speaker per box: gastro .082 → .125, plastic .333 → .395 (target .436), cardboard .317 (target .176). Speaker 4 in gastro is still the worst group in ALL (.125, miss .40).
- Unseen lasers: cardboard laser-shifted-3 .024 → .092, shifted-2 .323 → .383, plastic shifted-2 .238 → .267.
- The two-object regression from jitter is gone (.323 vs .328 without jitter), so the queued `--laser-jitter-p 0.5` arm was dropped: its reason no longer holds. The blur hypothesis wasn't tested directly. Augmenting the speaker likely made the model rely less on the exact spectral fingerprint, which full jitter had pushed it toward.
- Losses: broken −.082, cardboard cube-rotated −.118 (a 40-sample split).
- Why it works where the old `--color-aug` (4 cosines over 900 Hz) did nothing: that varied S(f) on a ~200 Hz scale, but measured speaker differences vary on a ~14–20 Hz scale (individual driver resonances), the scale the model's peak readers see.

### Laser-centering on top (s8-best-center): worse everywhere
- spkgain + `--laser-center 1`: subtract each capture's mean over the 100 lasers from every laser, in log power.
- @ep500 vs spkgain:
  - CORE .703 vs .722
  - HARD .586 vs .596
  - ALL .374 / .185 vs .388 / .168
  - unseen speaker .257 vs .275 (gastro speaker 4 .060 vs .125)
  - unseen laser .162 vs .201
  - lid .055 vs .096
- Why: the laser mean is not only the speaker's S(f). It also carries the object's *global* effect, its shift of every mode, which all lasers see. Centering throws that away, the per-sample version of the AdaBN failure. The speaker-gain augmentation already teaches invariance to the part that is S(f), without deleting the rest.

### Laser jitter (s8-best-jitter): better on CORE / HARD, but it hurts two-object scenes
- contour + `--laser-jitter 0.5`: each training sample's 10×10 wall field is resampled at a grid shifted by U(−0.5, 0.5) cells per axis (bilinear). This targets the probe result that the model reads exact laser placement (a 1-cell roll: .30 → .12).
- @ep500 vs contour:
  - CORE .705 / .009 vs .689 / .011
  - HARD .603 / **0 misses** vs .579 / .024; orientation 16.1° vs 17.6°
  - ALL .367 vs .355, but ALL miss .220 vs .202 (from the multi-object splits, below)
- By category (iou@.08; target / contour / jitter):

  | category | target | contour | jitter |
  |---|---|---|---|
  | unseen laser | .110 | .113 | **.170** |
  | unseen speaker | .260 | .199 | .228 |
  | broken | .190 | .474 | .577 |
  | unseen rotated object | .223 | .367 | .398 |
  | unseen object | .508 | .538 | .565 |
  | rotated | .409 | .532 | .564 |
  | **multiple** | .267 | **.328** | **.251** |

- Unseen lasers: cardboard vase-laser-shifted-2 .137 → .323, plastic cube-shifted-lasers-2 .141 → .238, backside .047 → .080. Larger shifts (cardboard shifted-3/-4) stay near 0: they move the grid beyond ±0.5 cell, or change the wall region it covers.
- Unseen speakers: cardboard .336 → .317, gastro .036 → .082, plastic .307 → .333 (target plastic .436). Better, but still below the target on average.
- CORE: biggest win plastic cube +.041; the cardboard random-poses layout +.086; the gastro far-wall/left region +.072.
- **Failure mode, multi-object (cube-vase, two-cubes, all 3 boxes):** iou falls in all 6 cells, at both ep300 and ep500. Misses rise (two-cubes .05 → .29 cardboard, .05 → .26 gastro), and so does CoM error (.14 → .20 cardboard cube-vase).
  - Hypothesis: a sub-cell bilinear shift is also a spatial low-pass filter (a half-cell shift averages two neighbours). A two-object scene is a superposition, and separating it needs the fine laser-to-laser pattern, which the blur removes. A single object only needs the coarse pattern.
  - Queued: `--laser-jitter-p 0.5` (s8-best-jitterp), which jitters only half the samples, so the model also learns from the sharp grid.

### The pose branch fixed rotation (s8-hybrid-posesum)
- The hybrid + a pose-factored branch (STN): a canonical object-centred mask, rotated by a predicted doubled angle (supervised by the mask's principal axis), placed at the **soft-argmax of the direct decoder's mask**, merged as a **residual** (direct + pose logits, zero-init).
- vs the hybrid @ep500:
  - CORE .662 / .025 vs .643 / .029
  - HARD .550 / .008 vs .518 / .016
  - **HARD orientation 21.3° vs 30.2°**
  - CORE boundary IoU .442 vs .413
- Unlike the hybrid, the rotation gain holds to the end of training (ep100 22.1°, ep500 21.3°).
- Rotated soap by layout:
  - rotated 25.5° → **16.9°**, rotated-2 30.8° → **16.6°**. That matches or beats the linear ridge probe on the raw spectrum (16.9° / 20.8°), so the model now uses the rotation information that is in the signal.
  - Correlation with the true angle: cos 2θ .27 → .44, sin 2θ .39 → .73.
  - Sideways stays wrong (61°, 12 samples): few training captures in that pose, a data limit.
- Why it works: rotation is one shared parameter learned from every rotated object, instead of a separate template per object and pose, and the residual merge lets it both add and erase pixels.
- Two failures led here: the soft-OR merge (over-painting, phantoms) and com-head centres (2× worse than the mask's own centroid).

### What the hybrid does better, and where it still fails

`src8/decompose.py`: IoU split into position / size / shape (CORE / HARD).

| run | IoU | CoM err | IoU, position fixed | + size fixed | orient |
|---|---|---|---|---|---|
| target (conv, log power) | .580 / .420 | .031 / .046 | .747 / .527 | .808 / .566 | 14.0° / 36.3° |
| conv, magnitude | .624 / .431 | .028 / .040 | .770 / .579 | .834 / .614 | 11.9° / 32.2° |
| hybrid | .646 / .518 | .024 / .033 | .772 / .603 | .835 / .645 | 11.6° / 30.2° |

- On CORE the gain is **localization**: CoM error is 23% lower. Shape is no better than conv on magnitude.
- Position is still the biggest CORE loss: .646 → .772 with exact centroids (~1.5 px at 64×64).
- HARD is shape-limited: .645 even with position and size fixed, vs .835 on CORE.

`src8/analyze.py RUN@EP --ref RUN@EP` → `runs/<run>/analysis.md` (by object, layout, speaker, floor region, wall distance, train distance, size). For the hybrid:
- **The weakest CORE item is the cardboard cube**: .329 IoU, 22% miss, area .77 (target .184).
  - By layout: grid-1 .333, grid-2 .459, random-pos2 .225 (37% miss), random_poses .223 (44% miss, CoM error .073 = 3× the CORE mean).
  - The failure is under-painting plus misses: low confidence on a weak signal. The cardboard cube changes the spectrum least of all objects (nb99).
  - Cardboard's smallest-object quartile scores .333 vs .775 for the largest.
- **Light cylinders are invisible**: cardboard 100 g .000 (100% miss), 200 g .011, 500 g .025. Rayleigh: Δω ∝ added mass, so a physics limit at this SNR.
- **Shifted lasers fail**: cardboard .054, plastic .237. The learned laser identities point at the wrong wall spots.
- **Unseen speaker**: gastro .103, plastic .371 (target .436, the hybrid's biggest loss).
- Floor region: no strong side-of-box effect (worst gastro far/right .54, best plastic far/left .74).
- Distance to the nearest training position: every CORE eval sample is within 0.03 (dense grids). The ">0.15 → .03" collapse comes only from eval-only layouts, so it's confounded with object novelty.

### Interpretability: what the models read
- The architecture's own readouts are **near uniform**: band gates within ±1.5% of uniform, laser attention within ±10%. A following linear layer can rescale anything, so they aren't importance. The repo's earlier attribution work found the same (ablation ≫ attention).
- Faithful importance by occlusion (`src8/attribution.py`): one 49 Hz band or one laser replaced by its box's mean signal, iou@.08 drop on CORE + HARD.

| box | hybrid's most needed bands | total band drop, hybrid / conv |
|---|---|---|
| cardboard | 962, 666, 912, 617 Hz | .865 / .668 |
| gastro | 568, 272, 912, 765 Hz | .638 / .391 |
| plastic | 223, 272, 174, 125 Hz | .359 / .397 |

- **Each box carries position in a different band.** Cardboard uses the high end (damping smears its overlapping low modes, so only the sparser high resonances stay resolvable). Plastic uses the low end.
- **The hybrid reads more of the spectrum than conv** in cardboard and gastro, which is where it gains most. Conv's mean over frequency positions discarded band-specific information that the gated per-band tokens keep.
- **No single laser matters** (≤ .012): the position information is spread redundantly over the grid.
- **Recording:** the log sweep gives 600–1000 Hz the least energy, which is exactly cardboard's informative band. Slow the sweep there, or extend it toward the ~1150 Hz Nyquist margin.

### Two-object scenes: the model draws one object (336 eval captures, iou@0.08, thr .08; "found" = >10% of the object covered)
| run | both found | one found | neither | blobs / scene | phantom | union iou (cube+cube / cube+vase, mean over boxes) |
|---|---|---|---|---|---|---|
| s8-conv-logpower | .37 | .51 | .12 | 1.95 | .49 | .20 / .32 |
| s8-hybrid-base | **.45** | .47 | .07 | 1.87 | .31 | .27 / .39 |
| s8-hybrid-posesum | .38 | .51 | .11 | 1.59 | .21 | .24 / .39 |
| s8-posesum-contour | .37 | .58 | .05 | 1.52 | .09 | .26 / .41 |
| s8-best-jitter | **.01** | .82 | .17 | 1.01 | .10 | .18 / .33 |
| s8-best-spkgain (best) | .29 | .66 | .05 | 1.47 | .12 | .25 / .41 |

- Each step that cleaned up single objects pushed the model further toward drawing one object. The pose branch (one slot) and the contour losses each cut "both found". Laser jitter collapsed it completely: exactly one blob per scene, both objects found in 1% of scenes. Speaker gain recovered part of it.
- The two objects are **never merged** into one blob between them (merged = 0). The model picks one object, usually the bigger one (vase over cube; in cardboard red over green cube), and draws it at about the right total area (area ratio ≈ 1).
- **The embedding knows there are two.** The count head says 2 on 93–94% of two-object scenes in every run, including jitter. The com head also roughly places the missed object: error .12 for missed vs .07 for found (spkgain). So the decoder drops information the embedding carries.
- Why: the decoder has no per-object structure. The pose branch is one slot, and the direct map has learned to suppress everything outside the slot.
- Fix queued: **s8-best-slots** (`--pose-merge slots`). K = 2 pose slots, each with its own centre heatmap, canonical mask, angle and exists logit. They are matched to the objects by centre (as Mask2Former's set matching; with K = 2 that is just the better of 2 orders) and supervised per object: the slot mask vs its own instance mask, the heatmap vs that instance, exists BCE. The union is a soft max over the exists-gated slots.

### Are two objects additive? Superposition augmentation (`src8/additivity.py`, no training): no
Idea (from Rayleigh perturbation, Δω_k ∝ Σ m_i φ_k(x_i)²): synthesize two-object training scenes as `X0 + (X_a − X0_a) + (X_b − X0_b)`, mask = union.
Two-object captures are 1913 of 19119 samples (10%: gastro 605, plastic 516, cardboard 792; an earlier "~3% / 360" figure was wrong).
- 1375 of them matched: each object to the single capture of the same type and speaker with mask IoU ≥ .5 (median .67). X0 = the nearest-in-time empty box of the same speaker.
- Correlation of the real two-object deviation with the sum of the two single deviations (log power, all boxes):

| | r_sum | r_a / r_b (one object) | r_null (random singles elsewhere) | r_rep (repeat ceiling) | fve |
|---|---|---|---|---|---|
| all | .347 | .257 / .291 | .244 | .535 | −0.97 |
| plastic | .60–.65 | .39–.55 | .44–.51 | .68–.70 | −0.25 to −0.34 |
| gastro | .26–.34 | .18–.33 | .19–.25 | .50–.53 | −0.9 to −1.5 |
| cardboard | .11–.23 | .08–.24 | .08–.10 | .39–.42 | −1.2 to −1.4 |

  Magnitude and power are no better (r_sum .368 / .349).
- **The model test.** s8-best-spkgain, iou@0.08 vs the union mask:
  - real captures: .345 (eval), .498 (train);
  - synthetic log-power sums: .259 / .242;
  - power-domain sums: .170 / .163;
  - **object a alone: .290 / .265.** The synthetic scene scores no better than showing the model one of the objects.
  - Random-pair null: .017.
- Why it fails:
  - **(1) Noise doubles.** Each single deviation carries its own non-repeatable drift (the repeat ceiling is only .54), so summing two of them adds two noise terms. That is why fve < 0: the sum overshoots.
  - **(2) A large common-mode "any object" response.** r_null is already .24, and .44–.51 in plastic.
  - **(3) The first-order regime breaks in cardboard.** The sum there is no better than random singles (.11–.23 vs .08–.10).
  - Plastic is the only box near its ceiling.
- Verdict: **superposition augmentation is not viable on this data.** Multi-object training has to come from real captures. The object-query decoder stays on the list, trained on real two-object scenes only.

### Tried and didn't work
- **Object slots matched by centre only** (s8-mag-slots v1, killed at ep80, kept as `runs/s8-mag-slots-v1-killed`). It learned fast: at ep50 CORE .505 vs .365 and HARD .439 vs .161 for s8-best-mag at ep50. But it painted 2× the area on CORE and 8× on ALL, and the exists BCE stayed at .63 (chance .69). In a single-object scene both slots sit on the one object, the centre cost ties and the match flips at random, so neither slot can learn whether it is on. Both slots get drawn. Fix (s8-mag-slots2): DETR's matching cost, 5 × centre L1 − p(exists). Ties then resolve the same way every time and reinforce.
- **Superposition augmentation** (synthetic two-object scenes from single captures): rejected before training. The model scores synthetic sums no better than one object alone (.26 vs .29). See the additivity section above.
- Per-box FiLM on the frequency stem (s8-best-film): neutral. CORE .725 vs .722 and HARD .600 vs .596 (both within noise). Its target, the cardboard cube, didn't move (.532 vs .538). Unseen laser fell .201 → .178 and unseen speaker .275 → .263. Per-box filters give the model one more way to fit box-specific detail. The per-box input norm plus the box embedding already handle the box difference. Not adopted.
- Laser-centering (s8-best-center): removes the object's global mode shift along with S(f); −.019 CORE, −.018 unseen speaker.
- **Pose-factored branch with a soft-OR merge** (`--pose-branch 1`, STN; Jaderberg 2015). A canonical mask per object is warped by com + angle and merged with `logaddexp`.
  - It gives the **lowest miss of any model**: ALL .196, CORE .012.
  - But IoU drops (CORE .570, ALL .295, below the target) and orientation doesn't improve (HARD 29.9° vs 30.0°).
  - Why: area ratio 1.5 (CORE) to 3.9 (ALL), phantoms .40 vs .22. An OR can only *add* pixels, so wherever the com is off it paints an extra blob the direct decoder can't erase, and the direct decoder still draws the shape, so rotation never has to go through the pose path.
  - Fix queued: `--pose-merge sum` (residual logits, zero-init: the identity at start; it can add and erase).
  - Second cause: the pose branch was placed at the **com head, which is 2× less accurate than the mask's own centroid** (CORE: hybrid head .043 vs mask .024; conv .054 vs .033). The pose run's mask centroid got worse (.030). This matches pose estimation: a dense map localizes better than global coordinate regression.
  - So the re-queued arm also uses `--pose-center mask`: one slot at the soft-argmax of the direct decoder's mask (integral regression, Sun et al. 2018).

- **AdaBN / test-time normalization** (Li 2016; TENT's statistics step, Wang 2021): re-estimate BatchNorm and box-norm stats on each unlabeled test domain (`src8/adabn.py`).
  - Unseen speaker: .370 → .068 (plastic), .102 → .016 (gastro), .208 → .076 (cardboard).
  - Shifted lasers: ≈ unchanged.
  - Cardboard CORE control: .606 → .436.
  - Why: a single-object domain's mean *is* the object's signature, so subtracting it removes the signal. Test-time stats would have to come from object-free (empty-box) captures of the new setup, which these splits don't have.

### Next arms from these failures
- **Tversky loss** (Salehi 2017; focal Tversky, Abraham 2019): FN weighted 0.7 vs FP 0.3, for the misses and under-painting of weak, small objects. `--loss-fn bce-tversky`, queued in v4.
- Ablations of the hybrid (v4): − box norm, − laser attention, magnitude input, to find what makes it localize better.



## 6 Multi-object decoders (2026-10-19 evening, the priority now)
Goal: draw any number of objects. The current best (s8-best-mag) has a one-slot pose branch, and its direct map learned to suppress everything outside that slot. Both objects are found in only .29–.36 of two-object scenes, although the embedding knows there are two: the count head is right on 93–94% of them (section above).
Superposition augmentation is out (see the additivity section). So every arm trains on the real two-object captures (two-cubes, cube-vase: 10% of the data, 80% of positions in train). All new arms use s8-best-mag's encoder and recipe minus the pose branch (`$NOPOSE`), so only the decoder differs. `src8/multi_object.py` scores them: both found, one, none, merged, phantoms, union iou.

| arm | decoder | why |
|---|---|---|
| s8-mag-nopose-r2 | direct conv decoder (one map, every object at once) | baseline: the decoder from before the pose branch, on the hybrid encoder |
| s8-conv-mag-r2 | the original conv-conv model, magnitude | baseline: the original architecture, every object at once |
| s8-best-mag | direct + 1 pose slot (sum merge) | the single-object best (trained already) |
| s8-mag-queries | Mask2Former object queries (`--obj-decoder queries`, K = 5, 3 layers) | the standard set decoder; any count ≤ K |
| s8-mag-centernet | centre heatmap + per-peak pose-branch mask + angle (`--obj-decoder centernet`) | keeps what fixed rotation (canonical shape + angle), once per object |
| s8-mag-slots2-r2 | 2 pose slots, DETR matching cost | the pose branch made two-object (fix of slots v1) |
| s8-mag-queries-os3 | queries + two-object scenes 3× per epoch (`--multi-oversample 3`) | the data side: 10% is little signal for "draw the second one" |
| s8-mag-query | the ModalFormer's own pixel-query decoder | a coordinate decoder (floor cells cross-attend to the lasers), every object at once |

- **queries** (`ObjectQueries`, arch.py). Each layer does three things: cross-attention to the 100 laser tokens; masked cross-attention to the trunk's 32×32 pixel features, so each query only looks where its own previous mask is on (Mask2Former's localization trick); then self-attention among the queries.
  - mask_k = ⟨mlp(q_k), pixel embedding⟩ (bias init −4, so 5 queries don't paint the union at start), plus an exists logit. Every layer is supervised.
  - Matching: cost = 5 bce + 5 dice − 2 p(exists) (Mask2Former's weights), trying every ordered choice of the queries (`match_sets`). Exists BCE gives unmatched queries weight .1 (DETR's eos_coef).
  - The mask is the soft union, logsumexp(mask_k + log p(exists_k)). The direct head stays as an aux loss.
- **centernet** (`CenterObjects`). A class-agnostic centre heatmap with CornerNet's focal loss against σ = 1.5 px gaussians at the instance centroids.
  - Each object gets a vector (the trunk's features at its centre plus the embedding), decoded to a canonical mask and angle by a one-slot PoseBranch and warped there.
  - Training is teacher-forced at the true centres. Eval takes the top 5 peaks above .3 (3×3 max-pool NMS), refined by the 3×3 soft centroid.
- The crash wave of 16:14–19:00 (s8-query, LFLF, farina, linskip; DataLoader worker "CUDA error: initialization error"): the band-drop code had a data-dependent `if not bkeep.all()` in ModalFormer.forward. It forced a graph break and recompiles until dynamo's cache limit, in every run (band_drop or not). Only these runs show it (best-mag / spkgain: 0 recompile-limit hits). The multiply is now unconditional. The link between the recompiles and the worker abort is inferred, not proven.

### First multi-object results (23:15): the direct decoder without the pose branch finds both objects twice as often
Two-object eval scenes (`multi_object.py`, 336 captures) and the goal metrics (`score.py`, iou / miss @0.08):

| run | both found | one | none | union iou | phantom | CORE iou / miss | HARD iou / orient | ALL iou / miss |
|---|---|---|---|---|---|---|---|---|
| s8-best-mag (1 pose slot) | .36 | .61 | .03 | .345 | .12 | .738 / .005 | .597 / 17° | .399 / .163 |
| **s8-mag-nopose-r2** (direct decoder, same recipe) | **.71** | .27 | .02 | **.413** | .19 | .738 / .006 | .580 / 27° | **.412** / .170 |
| s8-conv-mag-r2 (original conv-conv) | .45 | .43 | .13 | .275 | .46 | .626 / .031 | .434 / 36° | .328 / .234 |
| s8-conv-logpower | .37 | .51 | .13 | .254 | .49 | — | — | — |
| target s7-gpc-tukey-logpower-1k | — | — | — | — | — | .581 / .043 | .413 / 36° | .299 / .275 |

- **The pose branch was what made the model draw one object.** Without it, the same encoder and losses find both objects in .71 of scenes (from .36), with union iou .413 (from .345). CORE is unchanged (.738). ALL rises slightly (.412).
- What the pose branch buys is rotation only: HARD .597 vs .580, and orientation 17° vs 27° on the rotated soap.
- So the multi-object decoders (queries, centernet) are now competing with .71, not .36. They have to keep the pose branch's rotation gain without its one-object prior.
- The original conv-conv model finds both in .45 (more than the pose model) but paints phantoms in .46 of scenes and misses 13%.
- queries v1 OOMed at step 1 (512-sample microbatches); centernet v1 crashed in the explain callback (read `inst` in eval). Both are fixed and requeued (`-r2`, queries at microbatch 256).

### Re-plan (23:20): any number of objects AND rotation
- **Incident:** at 23:12 a global OOM killed the tmux train session, queue 5 and s8-mag-slots2-r2 (at ep293). superposition.py (~20 GB of RAM) ran alongside a run's dataset build. Separately, the disk was full: 184 GB of `experiments/mds` copies left behind by SIGKILLed runs, which skip their atexit cleanup. Cleared. The analysis now runs under `systemd-run --scope -p MemoryMax=18G`.
- **Physics for the design:** to first order an object shifts mode k by ∝ m φ_k(centroid)², and the next term is the Hessian of φ_k² contracted with the object's mass covariance. So an object's spectral signature is mostly (centroid, covariance = size + orientation). That makes centroid + covariance + shape residual the natural per-object representation, and rotation should be supervised per object through its covariance.
- New in arch.py:
  - `--imoment-weight`, `instance_moment_loss`: every pixel goes to the true object nearest in Mahalanobis distance (a gaussian mixture's assignment), and each object's predicted area, centroid and covariance in its cell is matched to its own. Every scene, any count. It equals the old moment loss on a single object.
  - `--obj-pose 1` (queries): each query also decodes a canonical mask and an angle (PoseBranch, one slot per query), warped to the soft centroid of the query's own mask and added to it (residual, zero-init). Its angle is supervised against its matched object (`obj_orient`, at the orient weight).
  - CenterNet gets the same per-object angle loss.
- queue 6 (`scripts/modal_src8_queue6.sh`, tmux train:src8-queue6). Every new arm is MULTI = the best recipe minus the pose branch, moment loss → per-instance moment loss:
  1. **s8-mag-qpose**: queries + per-query pose. The main candidate: count from set matching, rotation from the pose branch.
  2. **s8-mag-nopose-imom**: the direct decoder + the per-instance moment loss. Does the loss alone bring rotation back to the .71 both-found model?
  3. **s8-mag-centernet-r2**.
  4. **s8-mag-queries-r2**: queries without the pose head, to isolate the pose head.
  5. **s8-mag-qpose-os3**: qpose with two-object scenes 3× per epoch.
  6. **s8-mag-slots2-r3**: capped at 2 objects, so last.
  - Then the old queue-5 work (importance, mag-drop, mag-fjit, pixel-query, reruns).
  - Batch 1024 in microbatches of 512 (qpose peaks at 11.0 GB, smoke-tested).
- **qpose v1 failed (killed at ep57): a union-gating bug, now fixed.** At ep50, CORE .057, phantom .997, no object found in .94 of two-object scenes, although every training loss fell normally.
  - The union was logsumexp(mask_k + log p(exists_k)). A query with p(exists) = .01 only loses 4.6 logits, so a confident blob (logit +10) still painted.
  - The queries had split into two groups. Queries 2–3 drew ~125-px blobs with p .01 (the phantoms). Query 4 had p .99 and an empty mask. It was the one matched and supervised, and its pose head had no centroid to warp to.
  - Fix: `soft_or`, the probabilistic OR p = 1 − Π_k (1 − σ(mask_k)·σ(exists_k)), so a "no object" query adds at most its own p. The same bug was in `--pose-merge slots` (`with_slots`), which explains slots v1's 2× / 8× painted area; it uses soft_or now too.
  - Rerun as s8-mag-qpose-v2.
- Infrastructure:
  - The query arms OOM at microbatch 512 at the first training step, after the pre-training eval (11 GB in a bare smoke test). They run at 256 (gradient accumulation, the same batch of 1024).
  - A tmux server started after the OOM had `ulimit -n` 1024, so eval workers died with "Too many open files". Queues now set 524288.
- **Pending (CPU, deferred):** the linearity test with nonlinear competitors (`superposition.py`: inter = diag + per-feature v1⊙v2 interaction; rbf = kernel ridge on lowrank's coefficients). It was validated on toy data (interacting truth: inter .995 vs diag .879), but the real run was OOM-killed twice next to training.
- **qpose-v2 → v3 (00:50) → qpeak (01:55).** Debugged on a toy (`toy_queries.py` in the session scratchpad): 1–2 squares on a 32×32 floor, encoded by which lasers carry a bump. Both found / iou:
  - direct 1.00 / .646;
  - queries 1.00 / .659;
  - + pose head .16 / .588. The pose head's centre (the query mask's centroid) received the shape loss's gradient.
  - + detached soft-argmax centre 1.00 / .647 → **s8-mag-qpose-v3**;
  - + two-stage init (queries start at the direct map's top-5 local maxima) 1.00 / **.672** → **s8-mag-qpeak**, queued next.
  - v3 at ep50 on the real data: CORE .284 (baselines .37 at ep50). All 5 queries still sit at the mean position between the two objects: the slow DETR-style start that two-stage init addresses.
- 01:20 global OOM #2: ~70 eval loaders × 4 persistent workers (~280 processes, 42 GB). Eval loaders now have no workers. nopose-imom (lost at ep229) is requeued as -r2.

### s8-mag-qpose-v3: the new best on the goal metrics, and it draws several objects (ep500)
| run | CORE iou / miss | HARD iou / miss | rotated soap | ALL iou / miss | both found | union iou | phantom |
|---|---|---|---|---|---|---|---|
| s8-best-mag (1 pose slot) | .738 / .005 | .597 / 0 | **17.1°** | .399 / .163 | .36 | .345 | .12 |
| s8-mag-nopose-r2 (direct) | .738 / .006 | .580 / 0 | 27.4° | .412 / .170 | **.71** | **.413** | .19 |
| s8-conv-mag-r2 (conv-conv) | .626 / .031 | .434 / .048 | 35.5° | .327 / .234 | .45 | .275 | .46 |
| **s8-mag-qpose-v3** (queries + per-query pose) | **.745** / .006 | **.599** / 0 | 25.8° | **.415** / .182 | .61 | .404 | .19 |
| target s7-gpc-tukey-logpower-1k | .581 / .043 | .413 / .040 | 36.4° | .299 / .275 | | | |

- Object queries (5, set-matched) with a pose head per query: the best CORE, HARD and ALL of any model, and both objects found in .61 of two-object scenes (best-mag .36). It warmed up slowly: CORE .28 at ep50 (baselines .37), all queries at the mean position; then .735 at ep200.
- **Rotation is the remaining gap** (25.8° vs 17.1°). Its per-query angle loss reaches ~0 on train (.0013, like best-mag's .0012), so it fits rotation but does not generalize it. The free-form half of each query mask can memorize rotated templates.
- Next: s8-mag-qpeak (two-stage init, running); s8-mag-qonly (each query mask = its warped canonical alone, so rotation goes only through the angle; toy .548 / .80 vs v3 .647 / 1.00).

### Queue 6 results (ep500, scored 14:20): the multi-object arms tie on the goal metrics; none closes the rotation gap
| run | CORE iou / miss | HARD iou | rotated soap | ALL iou / miss | both found | union iou | phantom |
|---|---|---|---|---|---|---|---|
| s8-best-mag (1 pose slot) | .738 / .005 | .597 | **17.1°** | .399 / .163 | .36 | .345 | .12 |
| s8-mag-nopose-r2 (direct) | .738 / .006 | .580 | 27.4° | .412 / .170 | .71 | .413 | .19 |
| s8-mag-qpose-v3 (queries + per-query pose) | **.745** / .006 | **.599** | 25.8° | **.415** / .182 | .61 | .404 | .19 |
| s8-mag-qpeak (qpose-v3 + peak init) | .739 / .004 | .596 | 23.2° | .407 / .185 | .60 | .386 | .27 |
| s8-mag-qonly (query mask = warped canonical only) | .728 / .007 | .564 | 26.2° | .389 / .195 | .52 | .315 | .35 |
| **s8-mag-queries-r2** (queries, no pose head) | .731 / .005 | .598 | 26.9° | .412 / .179 | .73 | **.434** | .13 |
| **s8-mag-nopose-imom-r2** (direct + per-instance moments) | .742 / **.002** | .577 | 26.1° | .413 / **.164** | **.78** | .417 | .14 |
| s8-mag-centernet-r2 | .442 / .354 | .284 | 24.8° | .223 / .573 | .06 | .171 | .05 |

- Every multi-object arm lands at 23–27° on the rotated soap. Only the single global pose slot reaches 17°. Per-query pose heads, peak init and forcing rotation through the angle (qonly) all move it by at most 4°.
- **The per-query pose head costs counting**: queries without it find both objects in .73 of scenes (qpose-v3 .61), with the best union iou (.434), for about the same HARD.
- **The per-instance moment loss helps the direct decoder count**: both found .71 → .78, CORE miss .006 → .002. It does not help rotation (26°).
- Peak init (qpeak) fixed the slow start but adds phantoms (.27) and doesn't find more pairs. qonly is worse everywhere: a pure template is too rigid.
- CenterNet fails by under-detecting: CORE miss .35, and it finds neither object in .49 of two-object scenes. Its heatmap rarely clears the .3 peak threshold at eval (teacher-forced centres in training, so the mask path never sees its own peaks).
- So the rotation gain of best-mag looks tied to its one-object prior (one global canonical + angle fitted to the whole map), not to anything a per-object head learns. The rotated soap is a single-object scene, so the prior is right there.

### Ranking on the six things we want (`src8/rank.py`, ep500, 14:30)
Columns: dist = the four base objects × 3 boxes; both / union = two-object scenes; worst_box = the worst box's base iou; rot = cardboard rotated soap (iou, °); spk / laser = unseen speaker / shifted lasers; unseen_rot = sideways soap (gastro, plastic) + sideways charger; unseen_obj = coffee pot, biscoff, tomato sauce, charger; light = cylinder 100 g / 200 g, glass cup. rank = mean rank over these columns.

| run | dist | both | union | worst box | rot iou | rot ° | spk | laser | unseen rot | unseen obj | light | rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s8-mag-qpose-v3 | **.745** | .61 | .404 | .721 | **.599** | 25.8 | **.397** | .235 | .438 | .584 | .081 | **3.36** |
| s8-mag-queries-r2 | .731 | .73 | **.434** | .712 | .598 | 26.9 | .372 | .227 | .453 | .591 | **.089** | 3.45 |
| s8-mag-nopose-imom-r2 | .742 | **.78** | .417 | .714 | .577 | 26.1 | .376 | .217 | **.465** | .592 | .080 | 3.45 |
| s8-mag-nopose-r2 | .738 | .71 | .413 | .716 | .580 | 27.4 | **.397** | .215 | .456 | **.599** | .086 | 3.64 |
| s8-mag-qpeak | .739 | .60 | .386 | .712 | .596 | 23.2 | .378 | **.236** | .443 | .587 | .082 | 3.82 |
| s8-best-mag | .738 | .36 | .345 | **.722** | .597 | **17.1** | .368 | .212 | .452 | .585 | .080 | 4.64 |
| s8-mag-qonly | .728 | .52 | .315 | .697 | .564 | 26.2 | .366 | .226 | .444 | .585 | .077 | 6.27 |
| s8-conv-mag-r2 | .626 | .45 | .275 | .547 | .434 | 35.5 | .342 | .162 | .406 | .538 | .071 | 8.00 |
| target s7 | .580 | .36 | .267 | .511 | .420 | 36.3 | .260 | .110 | .377 | .508 | .045 | 9.00 |
| s8-mag-centernet-r2 | .442 | .06 | .171 | .364 | .283 | 24.8 | .223 | .087 | .317 | .415 | .027 | 9.36 |

- The top three are within .1 of each other in mean rank. On the single-object columns they differ by ≤ .02 iou; they separate only on counting (both .61 / .73 / .78) and the rotated soap angle.
- Light objects (≤ 200 g, glass cup) fail for every model (≤ .09). Unseen lasers (≤ .24) are the next weakest. Neither depends on the decoder: they're encoder / signal problems.
- queries-r2 is not simpler than nopose-imom-r2. Both train with the same losses (MULTI, imoment included); queries-r2 adds the query decoder, set matching and the exists loss on top. Its advantage is interpretability: one mask + p(exists) per object.

### Unseen rotation: HARD never tested it, so a 30-60° wedge is held out (`--dataset gastronorm_plastic_cardboard_wedge`)
- The rotated soap's mask angles (principal axis of `03_smask`) span 0-180°, and every HARD eval sample has a train sample within 0.7° on average (max 5.3°). HARD tests unseen positions at trained angles.
- The footprint is symmetric: iou with itself rotated 180° .947, mirrored about its long / short axis .94 / .94. So its angle is only defined mod 180° (and the mask can't tell the pump's side).
- The 60-120° wedge first proposed is not unseen: every upright soap grid (all 3 boxes, ~600 captures) sits at 75-105°, and the charger too. 30-60° is clear of upright (90°) and sideways (~0°). That's 27 positions (9 rotated, 18 rotated-2); a wedge sample's nearest kept angle is 8.6° away on average (max 15°).
- `dataset.UNSEEN_ANGLE` lists them. The `_wedge` dataset drops them from soap-dispenser-rotated (128 positions left, 26 in eval) into a new eval-only split `cardboard/soap-dispenser-unseen-angle` (category unseen-angle). Verified: 0 wedge positions in train.
- wandb: per split as always (`metrics/eval/cardboard/soap-dispenser-unseen-angle/{iou08,orient_err,...}`) plus `metrics/eval/unseen_angle/*`. viz finds splits from the outputs_history dirs, so it shows up there with no change.
- score.py adds two rows: WEDGE (the unseen split in a `_wedge` run; in older runs, their 6 eval positions inside the wedge, at trained angles) and HARD-kept (the rotated soap outside the wedge, the same angles in both).

  | run (ep500, original split: wedge angles trained) | HARD-kept iou / ° | WEDGE (6 eval positions) iou / ° |
  |---|---|---|
  | s8-best-mag | .594 / 18.2 | .610 / 12.6 |
  | s8-mag-queries-r2 | .606 / 27.6 | .568 / 24.5 |
  | s8-mag-nopose-imom-r2 | .584 / 27.1 | .548 / 22.3 |
  | s8-mag-qpose-v3 | .613 / 27.3 | .539 / 19.5 |

### New arms: the gaussian object decoder and CenterNet r3
- **`--obj-shape gauss`** (queries). Each query is an object: centroid μ, principal angle θ (a doubled-angle vector), two std devs (σ1, σ2), and a 16×16 shape canvas.
  - Its mask logit = 4 − |x'|² + canvas(x'), with x' = diag(1/σ) Rθᵀ(p − μ) the object's own frame. A uniform ellipse with std σ has its edge at |x'| = 2, so the prior alone is the moment-matched ellipse, and the canvas (zero-init, spanning |x'| ≤ 3) adds the shape.
  - Rotation is one explicit number per object, and the canvas is learned once for every angle.
  - `obj_gauss` loss (weight = --imoment-weight): the matched object's true centroid and covariance vs the query's, L1 (×100 on the covariance, as the moment loss). obj_orient supervises the angle vector.
  - Unit test: the thresholded rendered ellipse has exactly the query's centroid, angle (39.9° for 40°) and covariance.
  - Simpler than qpose: no pixel-embedding dot product, no separate pose branch, no soft-argmax centre.
- **CenterNet r3**: `--obj-count-floor 1` (eval keeps at least the count head's argmax best peaks, as well as any above .3), `--obj-heat-sigma obj` (heat gaussians = each object's covariance / 4 + 1 px², sized and oriented as the object), `--obj-jitter 1.5` (teacher-forced centres jittered 1.5 px, so the masks learn to draw from an imperfect peak).
- Toy check (`toy_queries.py`, 400 steps, CPU), iou / both found: queries .646 / 1.00, **qgauss .651 / 1.00**, centernet .634 / 1.00, centernet r3 .620 / 1.00. The gaussian decoder trains as well as plain queries. CenterNet works on the toy, so its real-data failure (no peak above .3 at unseen positions) doesn't reproduce there; r3 runs anyway, since the count floor targets exactly that failure.

### The wedge split is the default from now on (15:00)
- Comparability, verified by building both splits: every eval split except the rotated soap is identical (same positions and speakers). The rotated soap's eval positions are drawn as before and then the wedge is removed (`shuffle_keep` in build_splits), so the new HARD is exactly the old HARD's 25 positions outside the wedge. Compare a wedge run's HARD with an old run's **HARD-kept** (score.py). On wandb, `metrics/eval/hard/*` for a wedge run = those 25 positions.
- ALL / ALL-ood (score.py) and wandb's per-box `ood` leave the unseen-angle split out (it is NOT_OOD), so they stay comparable; it is reported on its own (WEDGE, `metrics/eval/unseen_angle/*`).
- Training loses 84 of 12080 samples (the wedge's 21 training positions × 4 speakers): 17% of the rotated-soap training positions, so a wedge run may lose a little on HARD.
- What it still doesn't test: one object (the soap), one box (cardboard), and only 27 positions (108 samples), so WEDGE is noisy. The cube's footprint is square, so its angle isn't measurable from moments. The wedge is a 30° hole (max 15° from a trained angle): it tests interpolating across a gap, not extrapolating to angles far outside training.
- queue 6 runs after the wedge block have their COMMON switched to the wedge dataset.

### s8-mag-qgauss-w (2026-10-20): an explicit Gaussian per query does NOT generalize in angle
ep500, wedge split (compare HARD-kept / WEDGE with the old runs' rows above):

| run | CORE | HARD-kept iou / ° | WEDGE iou / ° | both | union |
|---|---|---|---|---|---|
| qgauss-w | .712 / miss .009 | .583 / 32.4 | **.450 / 62.7** | .68 | .385 |
| queries-r2 (trained on the wedge) | .731 | .606 / 27.6 | .568 / 24.5 | .73 | .434 |
| best-mag (trained on the wedge) | .738 | .594 / 18.2 | .610 / 12.6 | .36 | |

- Worse everywhere, and on the unseen 30–60° wedge the angle error is 62.7°: it draws the soap at a trained angle, not between them. The explicit angle parameter (doubled-angle v from a Linear on the query) regresses the angle as a lookup, with no interpolation. A dense mask path (best-mag's canonical mask warped by the angle, or the plain direct head) degrades gracefully.
- So an interpretable shape prior bolted on as a REGRESSED parameter hurts. The fair wedge baselines (best-mag-w, queries-w, nopose-imom-w) are running next.

### Grayscale photo head (--gray, 2026-10-20)
- `gray = (1 − m)·bg[box] + m·app`: bg a learned floor per box (a parameter), app a 2-conv head on the trunk's pixel features, m the predicted mask **detached** (the photo loss never moves the mask, so mask metrics stay comparable; the shared trunk still gets the appearance gradient).
- The object is 2–5% of the image, so the loss is object-focused (`arch.gray_losses`): whole-image L1 of the composite + L1 of app on the TRUE object pixels, averaged over the object's own area (teacher-forced: appearance learns even where the mask is off) + L1 of image gradients within 2 px of the object (contour and inner detail stay sharp).
- Eval metrics: `gray_l1`, `gray_obj_l1` (composite on the true object pixels), `gray_bg_l1` (the empty floor drawn there: the bar app must beat). The OutputSaver saves gray_pred / gray_app for viz.
- Data check: floor brightness sd across photos is .003 / .002 / .010 (gastro / plastic / cardboard), so one learned floor per box is enough; objects are +.15–.30 brighter than the floor.
- Queued: s8-mag-gray-w (nopose-imom + --gray), then s8-mag-query-w (all-attention pixel-query decoder, MULTI recipe, moved up), after nopose-imom-w.

## Does the linear modal picture hold? (`src8/linear_physics.py`, nb110 to come; 2026-10-20)
Theory (first-order Rayleigh): an object adds mass, so each mode's frequency drops by δω_n/ω_n ≈ −½∫ρφ_n², linearly in the mass, and dy ≈ Jρ.
dy = log power minus the nearest-in-time empty box of the same speaker, in ~7.7 Hz bands.
- **dy vs the empty box is mostly session drift.** The cylinder sessions are 0.8–1.9 h from their empty box. In cardboard a cylinder's dy correlates .78 with a cylinder far away (>30% of the box) vs .80 at the same spot. Every test must remove each session's mean (per box, layout, speaker).
- **After removing each session's mean**, far pairs are at ≈0 and same-spot pairs at .15–.59: a real position-specific signal.
- **Not linear in mass** (cylinders 100/200/500/1000 g, same spot): in plastic, corr 100–200 .59, 100–500 .44, 100–1000 .30, 500–1000 .56. A linear model only rescales the pattern, so all of these would be equal. Its size grows ~25% (89 → 110) for 10× the mass (noise included).
- **Frequency shifts are tiny**: cross-correlation lag per band vs the empty box, over all lasers. They are all under one 0.77 Hz FFT bin.
  - gastro: −0.08 / −0.14 / −0.24 Hz at 100 / 500 / 1000 g, growing with frequency. This is mass loading as Rayleigh predicts (sublinear: 3× for 10× the mass).
  - plastic: **up**, +0.17 → +0.47 Hz with mass. That would be added stiffness (e.g. the flexible floor pressed onto its supports), not mass. Time gaps to the empty box are equal across masses (0.76–0.84 h), so drift can't make the mass trend. It could still add a common offset; the empty-vs-empty control was recorded minutes apart, which is too close to bound it.
  - cardboard: noisy (sd 0.6–1.3 Hz), no clear sign.
  - So the shift is far below a peak's width. The information is in how the amplitude pattern across lasers changes (mode shape and excitation), not in the frequency shifts. This agrees with nb103 (shape > level > freq).
- **Drift is real**: two empty boxes recorded back to back differ by |dy| 36–52, ~1 h apart by 83–105, and a candle vs its empty box by 107–166 (plastic 36 / 94 / 166). It's a per-session offset (maybe the box being reseated between sessions).
- **Forward ladder** (held-out positions, fve of dy after removing each session's mean, per box and speaker):

| box | object mean | + centroid (poly-4) | + covariance | J shared (mask, unit density) | J per object | kNN on the centroid |
|---|---|---|---|---|---|---|
| cardboard | .00 | .27 | .25 | .30 | .35 | .25 |
| gastro | .00 | .30 | .27 | .32 | .40 | .34 |
| plastic | .00 | .23 | .18 | .31 | .40 | .37 |

  - Covariance adds nothing, and hurts the soap: its effect is below what a linear model can pick out.
  - The full mask beats the moments (J per object +.10 over the centroid). Objects differ beyond "unit density": J per object > J shared.
- **Linear inverse** (box-normalized log spectrum → 64×64 mask, kernel ridge, threshold picked on a train slice), held-out IoU vs the network (nopose-imom-r2 @.08):

| | candle lin / rbf / net | cube | soap | vase |
|---|---|---|---|---|
| cardboard | .70 / .73 / .83 | .25 / .26 / .55 | .56 / .59 / .71 | .60 / .64 / .78 |
| gastro | .71 / .74 / .84 | .40 / .37 / .57 | .60 / .64 / .75 | .62 / .67 / .80 |
| plastic | .76 / .77 / .82 | .57 / .64 / .67 | .71 / .74 / .80 | .73 / .72 / .81 |

  - A purely linear map gets 85–93% of the network on the candle. The network's real gain is on the cube (cardboard .25 → .55) and in the weaker boxes.
- **Candle physics model** (`src8/candle_physics.py`): the box as K modes, the candle as a point mass at its centroid, solved exactly (Sherman–Morrison). It predicts log power as the measured box × the candle's physical ratio, because the phase isn't consistent between captures (coherence .1–.9 between two empties). Plastic, speaker 1, 202 train / 51 held-out:
  - K = 12: held-out fve .05, localization error .072 (random .38)
  - K = 24: .075 / .050
  - Bars: kNN on position .29; kNN in spectrum space .029
  - It underfits (train fve .10). Variants (finer floor basis, contact stiffness/damping) are running.
