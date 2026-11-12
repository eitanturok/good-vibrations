# viz

An interactive 3-D PCA/UMAP explorer of a run's input, latent and output vectors.

```sh
# latest checkpoint
.venv/bin/python figures/src/dimensionality_reduction/viz.py --run decres-rescale-aug10-v1 --port 8611

# a specific epoch
.venv/bin/python figures/src/dimensionality_reduction/viz.py --run decres-rescale-aug10-v1 --epoch 300 --port 8611
```

Then open http://localhost:8611. Features:

- filter by box, speaker, object
- vector: input, latent or output
- method: PCA or UMAP, 2-D or 3-D
- colour and size by any feature
- pick the axes, or the best components for a feature
- pick which train/eval splits are shown
- image size and density sliders
- title lists every setting
- settings saved in the URL
- download as SVG

```sh
# xy colour legend: dots on each box photo (overhead + 45 deg) + a plain legend, into box_position_legend_out/
.venv/bin/python figures/src/dimensionality_reduction/box_position_legend.py
```

## Final figure

`dimensionality_reduction.svg` (in this directory) is composed by hand (Photoshop) from the pieces in this directory:
the SVG downloads of the viz (`plastic_s1.svg`, `gastro_s7.svg`) and the legends in `box_position_legend_out/`.
No script writes the final figure.
