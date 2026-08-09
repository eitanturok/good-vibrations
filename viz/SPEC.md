# viz display spec

What identifies a thing at each display stage, and which coordinate system is authoritative
where. Written after a bug that rendered every prediction 10 rows off on gastronorm while
showing a correct IoU beside it.

## 0. Two coordinate systems

| | `SampleId` | `Row` |
|---|---|---|
| is | the sample directory name, as int (`"000010"` → `10`) | index into the ground-truth arrays |
| domain | sparse, arbitrary, **need not start at 0** | dense, `0 .. len(gt)-1` |
| authoritative for | the wire: URLs, JSON `s.i`, `.pt` `info["sample_id"]` | every numpy/torch array in the process |
| from | the client, `gt.sample_ids[row]`, `.pt` files | `_sid()` / `registry.sample_index()` |

`SampleId` is a sample's *external* name; `Row` is its *internal* one. They coincide only by
accident on `experiment-25`, whose ids happen to start at `000001`. On
`31_07_2026_gastronorm_exp1` ids start at `000009`, and `000009` itself is dropped for
having no downsampled mask, so `row = id - 10`. **Any code that assumes `row == id` is
correct on one dataset and silently wrong on the other.**

`RunData` adds a third space, **run-row**: dense over the ids one run actually predicted,
after dedup and after dropping ids this dataset has no target for. `RunData.row_of` is the
only bridge from `Row` to it.

## 1. The overhead image

Keyed by **`Row`**. The only sanctioned filesystem join is `Registry.sample_dir(row)`
(`data.py`), which reads the directory name out of `gt.sample_ids[row]` — so no
user-supplied string ever reaches a path. `render._backdrop(row)` takes a `Row`, and its
`lru_cache` is keyed on one; **passing an id there poisons the cache for the real row of
that number.**

The photo, never the grid, defines display geometry: `scene_aspect()` reads the aspect from
a real backdrop and `canvas_size()` sizes from the photo, flooring at `h * UPSCALE`.

## 2. The predicted mask

Two hops: `SampleId` —(`_sid`)→ `Row` —(`rd.row_of`)→ run-row.

`RunData.masks`, `.mse`, `.iou`, `.comdist`, `.com_pred`, `.splits` and `.sample_ids` all
share one row space, established by filtering them in lockstep in `load_run`.

`/api/frames` is the **one exception**: it stays in `SampleId` space end-to-end, because
`load_epoch_masks` matches raw ids read out of the `.pt`. It calls `sample_index()` only to
validate, and discards the result. Do not "fix" this.

The wire is `SampleId` everywhere — `/api/run/{name}` keys its per-sample metrics by id, to
match `/api/samples`.

## 3. The ground truth mask

Always **`Row`**. `gt.masks`, `gt.meta`, `gt.com_gt`, `gt.avg_com`, and every array from
`masks_at(shape)` / `com_at(shape)` share one row space.

`masks_at` guarantees this by **zero-filling** a sample that has no mask at the requested
size rather than dropping it — dropping would shift every later row out of step with
`gt.masks` and silently mis-pair predictions with targets.

A shape this dataset has no masks at raises, rather than falling back to the primary grid:
a silent fallback would put an unrelated resolution beside the runs.

## 4. Across mask sizes

Each run is scored at **its own** grid, against `gt.masks_at(run.shape)` — never against
the primary grid. A 16x16 and a 30x30 column can therefore sit side by side.

**Grid size never affects the id/row mapping.** A `Row` is valid at every size, which is
why resolution and sample identity are independent concerns.

Display geometry is likewise grid-independent: the cell box comes from `scene_aspect()`, and
the mask is stretched onto it anisotropically — which exactly undoes the anisotropic box
binning that produced the mask, landing every cell back on the pixels it averaged. So a
21x30 and a 20x40 mask of the same scene occupy the same box and align over the same photo.

## 5. The rule

**`_sid()` in `app.py` is the only `SampleId` → `Row` conversion, and it happens once per
request, at the HTTP edge. Below `app.py`, everything is a `Row`.**

A map named `row_of` is *keyed by* whatever `_sid()` produces. `GtIndex.row_of` is the sole
exception — it is keyed by `SampleId`, because it is the map that performs the conversion.
Two maps sharing that name with opposite key domains is exactly what caused the original
bug: `RunData.row_of` was built id-keyed while all four of its consumers passed rows.
