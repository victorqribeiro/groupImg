# Better image descriptors for groupImg

Which description of an image makes groupImg put similar-looking images
together. Clustering stayed as it was (squared L2, greedy k-means++, 10
restarts); only the feature vector changed.

**Conclusion: square-rooted color histograms of the whole image and of its
three horizontal bands, plus edge directions and texture per band.** In a
blind test, images' nearest neighbors looked better with edges and texture
than with color alone in 5 of the 6 cases where the two differed, and
labels agree (27.6% of the 8 nearest images share the class, against 18.1%
with the old histogram). The 12-group splits looked the same either way.
Reading an image costs what it did before, memory per image goes down, and
k-means got 3-5 times faster on the longer vectors.

## What changed in groupImg

In `cli/groupimg.py` and `gui/groupImgGUI.py`:

- **The feature vector** (`describe()`), 313 values, each part weighted 25%:
  - color: 64-bin RGB histogram (4 levels per channel) of the whole image
  - color: the same per horizontal third (sky above, ground below)
  - edges: direction of brightness changes in 8 bins plus a "flat" bin,
    per third. Without the flat bin a smooth image would be described by its
    JPEG noise.
  - texture: rotation-invariant local binary patterns (10 bins), per third
  - Every histogram is divided by its pixel count and square-rooted, so the
    squared distance between images is twice their Hellinger distance and
    one big background color can't outweigh the rest.
- **EXIF rotation.** Phone photos stored sideways are turned upright, which
  matters now that position (bands) counts.
- **Transparency** is put on white. Before, transparent areas counted as
  whatever color hid under them, often black.
- **`-s`** uses the log of the area and of the aspect ratio, weighted like
  one more part (`SIZE_WEIGHT`). Before, raw width and height (thousands)
  outweighed the colors (0-100) completely.
- **float32 vectors** instead of Python lists: 1.2 KB per image against
  about 2.1 KB before, despite 5 times as many values.
- **Faster k-means, same groups.** `distances()` recomputed every image's
  squared length and converted float32 data to float64 on each call, and
  `tolerance()` did the same again. Both now use a length computed once, and
  the multiplication runs in float32. The tie tolerance scales with the
  precision of the data (measured worst rounding: 4e-7, tolerance 7.6e-6).
  Seeding uses the same matrix form. A cap of 1000 passes guards against
  rounding keeping k-means busy.

## How it was tested

2,000 images (`make_mix.py`): 1,500 photos from
[Imagenette](https://github.com/fastai/imagenette) (150 from each of its 10
classes) and 500 textures from the
[Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
(50 from each of 10 random kinds). Victor judged blind pages
(`judge.py`, `page.html`, `serve.py`): for a query image, one row per option
with its 8 nearest images, in shuffled order, and each option's 12 groups.
Votes are in `results/`.

The labels (church, striped...) gave secondary scores, shown only after the
votes: the share of each image's 8 nearest images with the same class or
the same kind (photo or texture), and group purity (share of images whose
class is the most common in their group). Labels name objects, not looks,
so they favor whatever tracks objects; the votes came first.

### Round 1: color only (`results/round1.json`)

| option | values | same class in 8 nearest | same kind | group purity |
|---|---|---|---|---|
| old: 64-bin histogram in percent | 64 | 18.1% | 74.7% | 18.1% |
| square root (Hellinger) | 64 | 21.2% | 77.0% | 19.4% |
| square root + 2x2 quadrants | 320 | 21.6% | 74.8% | 20.8% |
| **square root + 3 bands** | 256 | **23.1%** | 75.9% | 20.5% |

The options really differ (the old histogram shares about 3 of 8 neighbors
with the others), but they looked equally good: of 8 answers, 3 were ties
and the rest went to all four options. Judging stopped there and the scores
decided. Most of the gain is the square root, which costs nothing.

### Round 2: adding edges and texture (`results/round2.json`)

12 query images, from the tenth of images where the two options' neighbors
differ most (at most 3 of 8 shared), so every question showed a difference.
Shared images were lined up at the start of both rows.

| | color (square root + bands) | + edges + texture |
|---|---|---|
| neighbor votes | 1 | **5** (6 ties) |
| groups | no real difference | no real difference |
| same class in 8 nearest | 23.1% | **27.6%** |
| same kind in 8 nearest | 75.9% | **81.3%** |
| group purity, k = 12 / 25 / 50 | 20.5 / 24.6 / 28.7% | 20.0 / 24.7 / **30.1%** |

Wins came from photos (French horn, gas pump) and textures (meshed,
striped). Splitting 2,000 mixed images into 12 groups is still decided by
broad color and brightness; the difference starts to show in finer groups.

### Weights and variants, by scores only

| variant | same class | same kind | purity k=12 | purity k=50 |
|---|---|---|---|---|
| **equal weights (shipped)** | 27.6% | 81.3% | 20.0% | 30.1% |
| color 70%, edges 15%, texture 15% | 25.9% | 79.9% | 21.3% | 29.3% |
| color 30%, edges 35%, texture 35% | 30.1% | 82.8% | 21.7% | 31.7% |
| no texture (40/35/25) | 25.7% | 79.6% | 20.7% | 28.6% |
| HSV bins instead of RGB | 27.9% | 81.2% | 21.3% | 29.5% |
| edges and texture of the whole image | 27.0% | 81.4% | 20.6% | 31.1% |

Less color scores higher, but the labels reward structure, and only the
50% split was confirmed by eye. `WEIGHTS` in the CLI changes it. HSV costs 5
times more than RGB for no gain.

### Speed

| | before | after |
|---|---|---|
| `read_image`, 12 MP photo / 12 MP noise | 12.4 / 70 ms | 12.4 / 71 ms |
| descriptor alone, from the 128 px thumbnail | 0.06 ms | 0.8 ms |
| k-means, 13,394 images, k = 5-20, 313 values | 11-28 s | 3-7.5 s |
| k-means, same images, old 64 values | 1.6-7.7 s | 0.7-3.6 s |
| `seeds.py` (5,000 synthetic images, k = 20) | 0.62 s, 10/10 found | 0.34 s, 10/10 found |

Over 18 runs (k = 5, 10, 20, 3 seeds, both feature sets) the new k-means'
total cost was within 0.001% of the old. 17 runs gave the same groups (ARI
0.95-1.0 between old and new); one (k = 20) found a different grouping with
a slightly lower cost. The CLI groups the 2,000 test images in 2.4 s
from start to end.

## Next: DCT descriptors (written, not run yet)

Three blocks built on the discrete cosine transform (the transform JPEG
uses) are in `descriptors.py`, to be scored before any of them reaches the
CLI:

- **`freq`, the frequency spectrum.** The gray thumbnail is cut into 8x8
  blocks as in JPEG. Each block splits one vote between 14 frequency bands
  (u + v of its coefficients) by its average coefficient size in each, or
  gives it to a "flat" bin when its pixels vary less than `FLAT_THRESHOLD`.
  This says how fine or coarse the detail is, wherever it is, which the
  current blocks can't tell: edges and texture only look 1 pixel around
  each pixel of the 128 px thumbnail. It is a histogram like the others
  (15 bins, `freq@bands3` per third). Expected to help the textures most.
- **`dct`, the layout.** The 63 lowest frequencies of the gray image
  squashed to 32x32, without the average, scaled to length 1: where light
  and dark are, regardless of brightness and contrast. The DCT keeps
  distances unchanged, so this is close to comparing tiny blurred
  thumbnails. Expected to be thrown off by framing (a church on the left vs
  the right) and to do poorly on class scores, but to match resized and
  re-saved copies of the same photo.
- **`phash`.** The same coefficients reduced to above or below their median,
  as the pHash perceptual hash does.

`scores.py` prints the label scores of the tables above for any configs,
without a judging page. `duplicates.py` asks a different question: in a
folder of recovered files, copies of the same photo should land together.
It copies 200 images in 5 ways (half size at JPEG quality 50, a 160 px
thumbnail, 10% cropped off each side, brighter with less contrast,
mirrored), then checks whether each copy's nearest original is its own and
whether k-means puts it in its original's group.

To run, after `make_mix.py` and `cache.py` as below:

```
F=benchmarks/features
BASE="rgb+rgb@bands3+edges@bands3+lbp@bands3"
python -I $F/scores.py "shipped=$BASE" \
    "+freq=$BASE+freq@bands3" "freq for lbp=rgb+rgb@bands3+edges@bands3+freq@bands3" \
    "+dct=$BASE+dct" "+phash=$BASE+phash" \
    "freq alone=freq@bands3" "dct alone=dct" "phash alone=phash"
python -I $F/duplicates.py $F/data/mix "shipped=$BASE" \
    "+freq=$BASE+freq@bands3" "+dct=$BASE+dct" "+phash=$BASE+phash" "dct alone=dct"
```

Adding a block at equal weight takes 1/5 of the distance from the others,
so a config that wins should be retried at other weights (`...+dct*0.5`)
before a blind round with `judge.py`. The thresholds were set by hand.

## Files

- `descriptors.py`: every block tried (and the DCT ones still to try), as configs such as
  `rgb+rgb@bands3+edges@bands3+lbp@bands3` (see its docstring).
  `describe()` in the CLI gives exactly what this config gives.
- `make_mix.py`: builds the test set from the two downloads.
- `cache.py`: decodes a folder once into 128 px thumbnails and 256 px copies
  for the pages.
- `scores.py`: label scores only, for shortlisting configs.
- `duplicates.py`: the near-duplicate test.
- `judge.py`: builds a blind round. `--pick disagree` asks only where the
  options differ.
- `page.html`, `serve.py`: the judging page and a server on 127.0.0.1 that
  saves votes to `results/`.

From the repo root, with Imagenette (`imagenette2-320.tgz`) and DTD
(`dtd-r1.0.1.tar.gz`) extracted under `benchmarks/features/data/`:

```
python -I benchmarks/features/make_mix.py IMAGENETTE_DIR DTD_DIR benchmarks/features/data/mix
python -I benchmarks/features/cache.py benchmarks/features/data/mix
python -I benchmarks/features/judge.py --round 2 --k 12 --queries 12 --pick disagree \
    "color=rgb+rgb@bands3" "color+edges+texture=rgb+rgb@bands3+edges@bands3+lbp@bands3"
python -I benchmarks/features/serve.py    # then open http://127.0.0.1:8765/round2/
```

## Caveats

- One judge, few votes: round 2 rests on 6 decided questions.
- The test set has photos and textures, not drawings, screenshots or icons,
  which the original use case also has. Image sizes barely vary, so the new
  `-s` and its weight weren't judged.
- The edge and texture thresholds were set by hand, not tuned. Grainy or
  heavily compressed images count as busier.
- Edges are measured on the 128 px thumbnail, so tiny images get different
  edge statistics from large ones and drift apart a little even without
  `-s`.
