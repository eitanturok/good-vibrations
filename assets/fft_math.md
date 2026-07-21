# FFT Representations & Spectral Metrics: Comprehensive Reference

This reference document provides a mathematical breakdown of fundamental spectral representations and derived metrics computed from the Discrete Fourier Transform (DFT / FFT).

## Notation & Conventions

- **Time domain index:** $t$ ($t = 0, 1, \dots, N-1$)
- **Frequency domain index:** $f$ ($f = 0, 1, \dots, N-1$)
- **Time domain signal:** $x[t]$
- **Frequency domain output:** $X[f]$
- **Sampling frequency:** $f_s$
- **Number of FFT points:** $N$
- **Complex Number Identity:** $X[f] = \text{Re}\{X[f]\} + j \cdot \text{Im}\{X[f]\} = |X[f]| e^{j \angle X[f]}$

---

## Metric Comparison Table

| Metric / Term | Math (Cartesian) | Math (Polar) | Meaning & Intuition | Primary Use Cases |
| :--- | :--- | :--- | :--- | :--- |
| **FFT ($X[f]$)** | $\text{Re}\{X[f]\} + j \cdot \text{Im}\{X[f]\}$ | $\|X[f]\| e^{j \angle X[f]}$ | Full complex output at frequency bin $f$ computed from time signal $x[t]$. | Fast convolution, spectral decomposition, filter design, inverse transform reconstruction. |
| **FFT Real** | $\text{Re}\{X[f]\} = \sum_{t=0}^{N-1} x[t] \cos\left(\frac{2\pi f t}{N}\right)$ | $\|X[f]\| \cos(\angle X[f])$ | In-phase (cosine) projection of the signal at frequency $f$. | Signal reconstruction, symmetric signal analysis, Hilbert transform calculations. |
| **FFT Imaginary** *(FFT "Image")* | $\text{Im}\{X[f]\} = -\sum_{t=0}^{N-1} x[t] \sin\left(\frac{2\pi f t}{N}\right)$ | $\|X[f]\| \sin(\angle X[f])$ | Quadrature (sine) projection of the signal at frequency $f$. | Phase angle calculation, asymmetric signal analysis, analytic signal generation. |
| **FFT Magnitude** | $\sqrt{\text{Re}\{X[f]\}^2 + \text{Im}\{X[f]\}^2}$ | $\|X[f]\|$ | Absolute amplitude strength at frequency $f$, independent of time shift or phase. | Audio equalization, spectral visualization, feature extraction (e.g., MFCCs). |
| **FFT Phase** | $\text{atan2}\Big(\text{Im}\{X[f]\},\, \text{Re}\{X[f]\}\Big)$ | $\angle X[f]$ | Phase angle (radians/degrees) representing time shift relative to $t=0$. | Acoustic beamforming, room impulse response, phase vocoders, time-delay estimation. |
| **Signal Energy ($E$)** | $\frac{1}{N}\sum_{f=0}^{N-1} \left(\text{Re}\{X[f]\}^2 + \text{Im}\{X[f]\}^2\right)$ | $\frac{1}{N}\sum_{f=0}^{N-1} \|X[f]\|^2$ | Total accumulated energy across all frequency bins (conserved via Parseval's Theorem: $\sum_{t=0}^{N-1} \|x[t]\|^2$). | Transient/pulse characterization, event detection, signal normalization. |
| **Average Power ($P$)** | $\frac{1}{N^2}\sum_{f=0}^{N-1} \left(\text{Re}\{X[f]\}^2 + \text{Im}\{X[f]\}^2\right)$ | $\frac{1}{N^2}\sum_{f=0}^{N-1} \|X[f]\|^2$ | Average energy per sample; rate of energy delivery for continuous or periodic signals. | Signal-to-Noise Ratio (SNR) calculation, link budgets, noise floor estimation. |
| **Power Spectral Density (PSD)** | $\frac{\text{Re}\{X[f]\}^2 + \text{Im}\{X[f]\}^2}{f_s \cdot N}$ | $\frac{\|X[f]\|^2}{f_s \cdot N}$ | Signal power normalized by frequency bin width ($\text{V}^2/\text{Hz}$ or $\text{W}/\text{Hz}$). | Random noise characterization, structural vibration analysis, comparing spectra across sample lengths. |

---

## Mathematical Insights & Transformations

### 1. Conversions Between Representations
- **Cartesian to Polar:**
  - Magnitude: $|X[f]| = \sqrt{\text{Re}\{X[f]\}^2 + \text{Im}\{X[f]\}^2}$
  - Phase: $\angle X[f] = \text{atan2}(\text{Im}\{X[f]\}, \text{Re}\{X[f]\})$
- **Polar to Cartesian:**
  - Real: $\text{Re}\{X[f]\} = |X[f]| \cos(\angle X[f])$
  - Imaginary: $\text{Im}\{X[f]\} = |X[f]| \sin(\angle X[f])$

### 2. Parseval's Energy Conservation Identity
The total energy summed in time equals the normalized total energy summed across frequency bins:
$$\sum_{t=0}^{N-1} |x[t]|^2 = \frac{1}{N} \sum_{f=0}^{N-1} |X[f]|^2$$

### 3. Magnitude vs. PSD Normalization
- **Raw FFT Magnitude $|X[f]|$:** Scales linearly with record length $N$. If $N$ doubles for a constant sinusoid, the raw magnitude peak doubles.
- **Power Spectral Density (PSD):** Normalizes for sampling rate $f_s$ and signal length $N$ (or window equivalent noise bandwidth $U$), rendering density in $\text{V}^2/\text{Hz}$ invariant to $N$.