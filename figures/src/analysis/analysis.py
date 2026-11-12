"""Three panels in two layouts, same panels and same text sizes in both:

  figures/figure_analysis.svg       one CVPR column (237 pt): equations, dimensionality reduction, frequency sensitivity stacked
  figures/figure_analysis_wide.svg  both columns (497 pt): dimensionality reduction left, equations over frequency sensitivity right

Every panel sits on a faint grey card; the bottom LABEL_SPACE pt of each card are left empty for a label added by hand.

  (a) latent arithmetic         pictures taken from figures/src/latent_tasks/both_equations.svg, laid out again here
  (b) dimensionality reduction  figures/src/dimensionality_reduction/dimensionality_reduction.svg, text resized
  (c) frequency sensitivity     drawn again here from the caches of figures/src/frequency_sensitivity

All words are FONT_SIZE pt, all math symbols (+ − = x y) are SYMBOL_SIZE pt, everything is Latin Modern Math.

    .venv/bin/python figures/src/analysis/analysis.py
"""
import base64
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src9").exists())
SRC = ROOT / "figures/src"
OUT = ROOT / "figures/figure_analysis.svg"
OUT_WIDE = ROOT / "figures/figure_analysis_wide.svg"

WIDTH = 237.0        # pt, one CVPR column (\linewidth)
FONT_SIZE = 8        # pt, every word in the figure
SYMBOL_SIZE = 10     # pt, math symbols: + − = and the x / y axis letters
LINE_HEIGHT = 9      # pt, between two lines of text
INK = "#1a1a1a"      # text colour (the dimensionality-reduction figure already uses it)
BOLD = f'stroke="{INK}" stroke-width=".45" stroke-linejoin="round"'   # Latin Modern Math has no bold face: thicken the glyphs with an outline, which every viewer draws the same
WIDTH_WIDE = 496.8   # pt, both CVPR columns (\textwidth)
LABEL_SPACE = 11     # pt, empty strip at the bottom of every card for a label added by hand
CARD = "#f4f6f9"     # faint cool grey behind every panel (a lighter shade of the walls of the 3-D plot)
CARD_EDGE = "#e1e5ec"  # hairline around a card
CARD_PAD = 3         # pt, from the edge of a card to its panel
CARD_GAP = 4         # pt, between two cards
FONT = "Latin Modern Math"

font_manager.fontManager.addfont(str(ROOT / "assets/fonts/latinmodern-math.otf"))


def text(x, y, string, size=FONT_SIZE, anchor="middle", extra=""):
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{INK}" {extra}>{string}</text>'


def card(x, y, width, height):
    """A grey card. Its bottom LABEL_SPACE pt stay empty, for a label added by hand."""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" rx="2.5" fill="{CARD}" stroke="{CARD_EDGE}" stroke-width=".5"/>'


def nest(svg, x, y, width):
    """Place a whole SVG file at (x, y), scaled to be width pt wide. Returns (svg text, height)."""
    start = svg.index("<svg")
    end = svg.index(">", start)
    view_box = re.search(r'viewBox="([^"]+)"', svg[start:end]).group(1)
    box_width, box_height = [float(v) for v in view_box.split()[2:]]
    height = box_height * width / box_width
    opening = f'<svg x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" viewBox="{view_box}"'
    return opening + svg[end:], height


# ---------------------------------------------------------------- (a) latent arithmetic
# One term = (title, picture, how to draw it, sign in front, coefficient).
# picture = index of the <image> in both_equations.svg (in file order), or None for the latent column.
EQUATIONS = [
    [("vase", 0, "photo", "", ""), ("mean vase", 1, "icon", "−", ""), ("mean soap", 2, "icon", "+", ""), ("decoded layout", 3, "mask", "=", "")],
    [("cube", 4, "photo", "", "half"), ("vase", 5, "photo", "+", "half"), ("mean object pair", None, "latent", "+", ""), ("decoded layout", 6, "mask", "=", "")],
]
OBJECT_AT = {0: (.405, .76), 4: (.435, .755), 5: (.67, .235)}   # picture index -> where its object is (x, y as fractions of the photo)
RING_RADIUS = .17    # in picture widths
RING_COLOUR = "#ffd21f"
LATENT_CELLS = ["#e3f1f3", "#7fbfca", "#e3f1f3", "#b9dde3", "#4f9fae", "#e3f1f3", "#b9dde3", "#7fbfca", "#e3f1f3", "#b9dde3"]  # as in figure_model_arch
PHOTO_WHITE = 99.5   # the photos are dark: this percentile of a photo becomes white ...
PHOTO_GAMMA = .55    # ... and the mid-tones are lifted with this gamma (1 = unchanged, smaller = brighter)
SIGN_SLOT = .2       # width of the + − = slot, in picture widths
COEF_SLOT = .26      # width of the ½ slot
EQUALS_EXTRA = .25   # extra width of the gap around =, so "mean object pair" and "decoded layout" do not touch
ROW_GAP = 5          # pt, between the two equations


def brighten(href):
    """A photo (PNG data URI) with its levels stretched and its mid-tones lifted."""
    png = base64.b64decode(href.split(",")[1])
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")) / 255
    pixels = np.clip(pixels / np.percentile(pixels, PHOTO_WHITE), 0, 1) ** PHOTO_GAMMA
    buffer = io.BytesIO()
    Image.fromarray((pixels * 255).round().astype(np.uint8)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def equations_panel(x0, y0, width):
    source = (SRC / "latent_tasks/both_equations.svg").read_text()
    pictures = [re.sub(r"\s", "", href) for href in re.findall(r'<image xlink:href="([^"]+)"', source)]
    assert len(pictures) == 7, "both_equations.svg changed: update EQUATIONS"

    # columns: [½ picture] [sign ½ picture] [sign ½ picture] [= picture]; the first sign slot is dropped.
    units = (COEF_SLOT + 1) + 3 * (SIGN_SLOT + COEF_SLOT + 1) + EQUALS_EXTRA
    side = width / units                 # a picture is side × side pt
    most_lines = max(title.count("\n") + 1 for terms in EQUATIONS for title, _, _, _, _ in terms)
    title_height = most_lines * LINE_HEIGHT + 1
    parts = []
    y = y0
    for terms in EQUATIONS:
        top = y + title_height           # top of the pictures
        middle = top + side / 2
        x = x0
        for i, (title, picture, kind, sign, coef) in enumerate(terms):
            sign_width = SIGN_SLOT * side if i > 0 else 0
            if sign == "=":
                sign_width += EQUALS_EXTRA * side
            coef_width = COEF_SLOT * side
            # the sign sits in its own slot when a ½ follows it, otherwise in the middle of the whole gap
            sign_x = x + sign_width / 2 if coef else x + (sign_width + coef_width) / 2
            if sign:
                parts.append(text(sign_x, middle + .25 * SYMBOL_SIZE, sign, SYMBOL_SIZE))
            if coef == "half":
                coef_x = x + sign_width + coef_width / 2
                parts.append(text(coef_x, middle - 2, "1"))
                parts.append(f'<line x1="{coef_x - 2.6:.2f}" x2="{coef_x + 2.6:.2f}" y1="{middle:.2f}" y2="{middle:.2f}" stroke="{INK}" stroke-width=".5"/>')
                parts.append(text(coef_x, middle + 7.2, "2"))
            left = x + sign_width + coef_width

            if kind == "latent":         # a column of shaded cells, as tall as a picture
                cell = side / len(LATENT_CELLS)
                cell_x = left + side / 2 - cell / 2
                for j, colour in enumerate(LATENT_CELLS):
                    parts.append(f'<rect x="{cell_x:.2f}" y="{top + j * cell:.2f}" width="{cell:.2f}" height="{cell:.2f}" fill="{colour}" stroke="#2f6f7a" stroke-width=".35"/>')
                parts.append(f'<rect x="{cell_x:.2f}" y="{top:.2f}" width="{cell:.2f}" height="{side:.2f}" fill="none" stroke="#1f4f57" stroke-width=".6"/>')
            else:
                fit = "xMidYMid meet" if kind == "icon" else "none"   # icons keep their shape, photos and masks fill the square
                href = brighten(pictures[picture]) if kind == "photo" else pictures[picture]
                flip = f"translate(0 {2 * top + side:.2f}) scale(1 -1)"     # matplotlib stores its images upside down
                parts.append(f'<image x="{left:.2f}" y="{top:.2f}" width="{side:.2f}" height="{side:.2f}" preserveAspectRatio="{fit}" transform="{flip}" xlink:href="{href}"/>')
            if kind == "photo":          # a ring so the small object is found at a glance
                ring_x = left + OBJECT_AT[picture][0] * side
                ring_y = top + OBJECT_AT[picture][1] * side
                parts.append(f'<circle cx="{ring_x:.2f}" cy="{ring_y:.2f}" r="{RING_RADIUS * side:.2f}" fill="none" stroke="{RING_COLOUR}" stroke-width=".9"/>')
            if kind == "mask":
                parts.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{side:.2f}" height="{side:.2f}" fill="none" stroke="#8c8c8c" stroke-width=".6"/>')

            lines = title.split("\n")    # the last line always sits just above the picture
            for k, line in enumerate(lines):
                baseline = top - 2.5 - (len(lines) - 1 - k) * LINE_HEIGHT
                parts.append(text(left + side / 2, baseline, line))
            x = left + side
        y = top + side + ROW_GAP
    return "\n".join(parts), y - ROW_GAP - y0


# ---------------------------------------------------------------- (b) dimensionality reduction
def dimred_panel(x0, y0, width):
    svg = (SRC / "dimensionality_reduction/dimensionality_reduction.svg").read_text()
    scale = width / 237.0                                      # the file is 237 pt wide; text is set so it ends up FONT_SIZE pt after scaling
    word = f"{FONT_SIZE / scale:.2f}"
    symbol = f"{SYMBOL_SIZE / scale:.2f}"
    for old in ("9.568000000000001", "10.4"):                  # legend entries, legend headings
        svg = svg.replace(f'font-size="{old}"', f'font-size="{word}"')
    for old in ("11.440000000000001", "13.520000000000001"):   # x / y axis letters
        svg = svg.replace(f'font-size="{old}"', f'font-size="{symbol}"')
    svg = svg.replace('dy="9.38"', f'dy="{8.2 / scale:.2f}"')   # second line of "empty box", "multiple objects"
    assert svg.count(f'font-size="{word}"') == 8 and svg.count(f'font-size="{symbol}"') == 8, "dimensionality_reduction.svg changed"
    assert svg.count('<tspan font-weight="bold">') == 2          # the "object" and "position" headings
    svg = svg.replace('<tspan font-weight="bold">', f'<tspan {BOLD}>')
    white = '<rect width="100%" height="100%" fill="#fff"/>'   # its white background would hide the card
    assert white in svg
    svg = svg.replace(white, "", 1)
    return nest(svg, x0, y0, width)


# ---------------------------------------------------------------- (c) frequency sensitivity
SET = "train"                                # samples behind the bars and the line
BOXES = ["plastic", "gastro", "cardboard"]   # rows, top to bottom
BOX_COLOUR = {"plastic": "tab:blue", "gastro": "tab:green", "cardboard": "tab:orange"}
BOX_CMAP = {"plastic": "Blues", "gastro": "Greens", "cardboard": "Oranges"}
FREQ_HZ = np.linspace(100, 1000, 1170)       # frequency of each FFT bin
BAND_HZ = 100 + 50 * np.arange(18) + 25      # centre of each 50 Hz band
LEGEND_HEIGHT = 24                           # pt, legend band on top of the panel: two lines of text


def frequency_panel(x0, y0, width, panel_height):
    drops = np.load(ROOT / f"runs/_src9_cache/nb136/q1_{SET}.npz")
    spectra = np.load(ROOT / f"runs/_src9_cache/frequency_sensitivity/spectra_{SET}.npz")   # built by freq_drop_sensitivity.ipynb
    assert (spectra["rows"] == drops["rows"]).all()

    style = {
        "font.family": FONT,
        "font.size": FONT_SIZE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": .5,
        "xtick.major.width": .5,
        "ytick.major.width": .5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",              # text stays text, like the other two panels
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width / 72, panel_height / 72))

        def fx(points):                      # pt from the left edge -> figure fraction
            return points / width

        def fy(points):                      # pt from the top edge -> figure fraction
            return 1 - points / panel_height

        left = fx(16)                        # room for the y tick labels
        right = fx(width - 10)               # half of the last tick label sticks out
        bottom = fy(panel_height - 11)       # room for the x tick labels
        top = fy(LEGEND_HEIGHT + 4)
        gap = 3 / panel_height
        row_height = (top - bottom - 2 * gap) / 3

        # legend, left part: what the line and the bars are
        line_y, bar_y = 5, 15                # pt from the top: middle of the first and of the second legend line
        squiggle_x = [2, 3.2, 4.2, 5, 5.8, 6.8, 7.6, 8.6, 9.6, 10.8, 12]         # pt: a tiny spectrum, one sharp peak then a bump and a tail
        squiggle_up = [-1.6, -.8, 0, 3, -.4, .9, .2, -.9, -.6, -1.4, -1.6]      # pt above the middle of the legend line
        fig.add_artist(Line2D([fx(x) for x in squiggle_x], [fy(line_y - up) for up in squiggle_up], color="k", lw=.6))
        fig.text(fx(15), fy(line_y), "FFT magnitude", va="center_baseline")
        for k, shade in enumerate([.35, .6, .85]):   # three little grey bars
            fig.patches.append(plt.Rectangle((fx(2.5 + 3.2 * k), fy(bar_y + 3)), fx(2.6), (2.5 + 2 * k) / panel_height,
                                             transform=fig.transFigure, color=plt.cm.Greys(shade)))
        fig.text(fx(15), fy(bar_y), "sensitivity (ΔIoU)", va="center_baseline")

        # legend, right part: one colour scale per box, lowest and highest ΔIoU of that box at its ends
        scale_left = 84                      # pt, where the three colour scales start
        slot = (width - 2 - scale_left) / 3
        number_width = 13                    # pt, room for ".01"
        for r, box in enumerate(BOXES):
            in_box = drops["box"] == box
            spec = spectra["spec"][in_box]
            sigma = spectra["sig"][in_box][:, None]
            spectrum = ((spec - spec.mean(1, keepdims=True)) / sigma).mean(0)   # the model's z-scored input, mean over samples
            drop = drops["mask_b"][in_box].mean(0)                              # mean ΔIoU when a band is masked

            slot_left = scale_left + r * slot
            slot_middle = slot_left + slot / 2
            scale_width = slot - 2 * number_width - 6
            fig.text(fx(slot_middle), fy(line_y), box, ha="center", va="center_baseline", color=BOX_COLOUR[box])
            scale_ax = fig.add_axes([fx(slot_middle - scale_width / 2), fy(bar_y + 2.5), fx(scale_width), 5 / panel_height])
            scale_ax.imshow(np.linspace(0, 1, 64)[None], cmap=BOX_CMAP[box], aspect="auto")
            scale_ax.set_xticks([])
            scale_ax.set_yticks([])
            for spine in scale_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(.4)
            lowest = f"{drop.min():.2f}".lstrip("0")
            highest = f"{drop.max():.2f}".lstrip("0")
            fig.text(fx(slot_middle - scale_width / 2 - 1.5), fy(bar_y), lowest, ha="right", va="center_baseline")
            fig.text(fx(slot_middle + scale_width / 2 + 1.5), fy(bar_y), highest, ha="left", va="center_baseline")

            row_bottom = top - (r + 1) * row_height - r * gap
            bar_ax = fig.add_axes([left, row_bottom, right - left, row_height])
            colours = plt.cm.ScalarMappable(plt.Normalize(drop.min(), drop.max()), BOX_CMAP[box])
            bar_ax.bar(BAND_HZ, drop, width=46, color=colours.to_rgba(drop))
            bar_ax.set_xlim(100, 1000)
            bar_ax.set_ylim(0, 1.18 * drop.max())
            bar_ax.axis("off")

            ax = fig.add_axes([left, row_bottom, right - left, row_height])     # the line, in front of the bars
            ax.patch.set_visible(False)
            ax.plot(FREQ_HZ, spectrum, color="k", lw=.6)
            margin = .05 * np.ptp(spectrum)
            ax.set_xlim(100, 1000)
            ax.set_ylim(spectrum.min() - margin, spectrum.max() + margin)
            high_tick = int(spectrum.max())          # two whole-number y ticks, never 0: e.g. 2 and 4, or 5 and 10
            if high_tick >= 10:
                high_tick = high_tick // 5 * 5
            ax.set_yticks([high_tick // 2, high_tick])
            ax.set_xticks(BAND_HZ[::4])
            if r < len(BOXES) - 1:
                ax.tick_params(bottom=False, labelbottom=False)
            else:                                    # the unit sits on the tick-label line, at its right end
                ax.annotate("Hz", (1, 0), xytext=(8, -4), xycoords="axes fraction", textcoords="offset points", ha="right", va="top")

        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", transparent=True)
        plt.close(fig)
    return nest(buffer.getvalue(), x0, y0, width)


# ---------------------------------------------------------------- the two layouts
def save(path, width, height, parts):
    header = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width:.1f}pt" height="{height:.1f}pt" '
              f'viewBox="0 0 {width:.1f} {height:.1f}" font-family="\'{FONT}\', \'Latin Modern Roman\', \'CMU Serif\', serif">')
    background = '<rect width="100%" height="100%" fill="#fff"/>'
    path.write_text("\n".join([header, background, *parts, "</svg>"]))
    print(f"saved {path}  ({width:.0f} x {height:.0f} pt = {width / 72:.2f} x {height / 72:.2f} in)")


CARD_EXTRA = 2 * CARD_PAD + LABEL_SPACE      # card height = panel height + CARD_EXTRA

# one column: three cards stacked. The dimensionality reduction is shrunk and centred to save height.
inner = WIDTH - 2 * CARD_PAD
DIMRED_WIDTH = 198                           # at this width the legend text fills its box exactly as in the hand-made file
FREQ_HEIGHT = 108
parts = []
y = 0
equations, height = equations_panel(CARD_PAD, y + CARD_PAD, inner)
parts += [card(0, y, WIDTH, height + CARD_EXTRA), equations]
y += height + CARD_EXTRA + CARD_GAP
dimred, height = dimred_panel((WIDTH - DIMRED_WIDTH) / 2, y + CARD_PAD, DIMRED_WIDTH)
parts += [card(0, y, WIDTH, height + CARD_EXTRA), dimred]
y += height + CARD_EXTRA + CARD_GAP
frequency, height = frequency_panel(CARD_PAD, y + CARD_PAD, inner, FREQ_HEIGHT)
parts += [card(0, y, WIDTH, height + CARD_EXTRA), frequency]
y += height + CARD_EXTRA
save(OUT, WIDTH, y, parts)

# both columns: the big card on the left, two short cards on the right, both sides equally tall
LEFT_WIDTH = 230                             # card of the dimensionality reduction; this sets the height of the whole figure
EQUATIONS_WIDTH = 232                        # narrower than its card: smaller pictures leave more height for the frequency panel
right_x = LEFT_WIDTH + CARD_GAP
right_width = WIDTH_WIDE - right_x
parts = []
dimred, dimred_height = dimred_panel(CARD_PAD, CARD_PAD, LEFT_WIDTH - 2 * CARD_PAD)
total = dimred_height + CARD_EXTRA
parts += [card(0, 0, LEFT_WIDTH, total), dimred]
equations, height = equations_panel(right_x + (right_width - EQUATIONS_WIDTH) / 2, CARD_PAD, EQUATIONS_WIDTH)
parts += [card(right_x, 0, right_width, height + CARD_EXTRA), equations]
y = height + CARD_EXTRA + CARD_GAP
frequency, _ = frequency_panel(right_x + CARD_PAD, y + CARD_PAD, right_width - 2 * CARD_PAD, total - y - CARD_EXTRA)
parts += [card(right_x, y, right_width, total - y), frequency]
save(OUT_WIDE, WIDTH_WIDE, total, parts)
