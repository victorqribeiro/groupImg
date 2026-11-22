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

Later a layout of light and dark that ignores mirroring was added, at 1/9
of the distance, so that brighter and mirrored copies of a photo land
together: see [Mirror-proof layout](#mirror-proof-layout-in-groupimg-not-judged-by-eye-yet).

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

## DCT descriptors and near-duplicates (scored, not judged yet)

Three blocks built on the discrete cosine transform (the transform JPEG
uses) are in `descriptors.py`. They were scored by labels and by a new
near-duplicate test, not by eye yet, and none of them is in the CLI.

- **`freq`, the frequency spectrum.** The gray thumbnail is cut into 8x8
  blocks as in JPEG. Each block splits one vote between 14 frequency bands
  (u + v of its coefficients) by its average coefficient size in each, or
  gives it to a "flat" bin when its pixels vary less than `FLAT_THRESHOLD`.
  This says how fine or coarse the detail is, wherever it is, which the
  current blocks can't tell: edges and texture only look 1 pixel around
  each pixel of the 128 px thumbnail. It is a histogram like the others
  (15 bins, `freq@bands3` per third).
- **`dct`, the layout.** The 63 lowest frequencies of the gray image
  squashed to 32x32, without the average, scaled to length 1: where light
  and dark are, regardless of brightness and contrast. The DCT keeps
  distances unchanged, so this is close to comparing tiny blurred
  thumbnails.
- **`phash`.** The same coefficients reduced to above or below their median,
  as the pHash perceptual hash does.

### Label scores (`scores.py`)

The scores of the tables above, same seed, each block added at equal weight
(1/5 of the distance) unless a weight is given:

| option | values | ms/image | same class | same kind | purity k=12 / 25 / 50 |
|---|---|---|---|---|---|
| shipped | 313 | 0.71 | 27.6% | 81.3% | 20.0 / 24.7 / 30.1% |
| + freq | 358 | 1.79 | **28.1%** | **82.4%** | 21.1 / 25.4 / **30.2%** |
| freq instead of texture | 328 | 1.58 | 27.1% | 81.5% | 20.8 / 25.3 / 29.9% |
| + dct | 376 | 0.79 | 24.5% | 81.0% | **23.2** / 25.9 / 28.4% |
| + dct at half weight | 376 | 0.80 | 26.5% | **82.4%** | 21.9 / 25.2 / 29.4% |
| + phash | 376 | 0.83 | 23.3% | 79.8% | 21.1 / **26.2** / 29.9% |
| freq alone | 45 | 1.12 | 18.4% | 78.9% | 19.1 / 21.3 / 24.0% |
| dct alone | 63 | 0.08 | 13.8% | 69.8% | 17.3 / 19.6 / 20.5% |
| phash alone | 63 | 0.11 | 9.2% | 68.1% | 12.8 / 14.4 / 16.2% |

Chance is 6.25% for the same class and 62.5% for the same kind. `freq` is
the only block that scores a little higher everywhere; it makes the
descriptor 2.5 times slower, still far below the 12 ms it takes to decode a
photo. `dct` and `phash` don't track classes, as expected of a layout: a
church on the left and one on the right have different layouts.

### Near-duplicates (`duplicates.py`)

This asks the question groupImg was first written for: in a folder of
recovered files, copies of the same photo should land together. It copies
200 images in 5 ways (half size at JPEG quality 50, a 160 px thumbnail, 10%
cropped off each side, 25% brighter with 15% less contrast, mirrored), then
checks whether each copy's nearest original is its own, and whether k-means
puts it in its original's group when it splits the 2,000 originals and
1,000 copies together. Every option finds the half-size copies and the
thumbnails (98-100%), so they are left out here:

| option | nearest is its own: cropped / brighter / mirrored | same group, k = 50: cropped / brighter / mirrored |
|---|---|---|
| shipped | 100 / **67** / 100% | 92 / **57** / 100% |
| + freq | 99 / 70 / 100% | 92 / 62 / 100% |
| + dct | 98 / 96 / **68**% | 88 / 82 / **82**% |
| + dct at half weight | 100 / 92 / 82% | 92 / 81 / 90% |
| + phash | 94 / 96 / 52% | 90 / 78 / 84% |
| dct alone | 76 / 100 / 19% | 66 / 99 / 40% |

Random groups of the same sizes would put 2-3% of copies with their
original. The shipped descriptor's weak spot is brightness and contrast,
likely because its color bins are fixed, so a brighter copy's colors move
to other bins all at once, and its fixed edge threshold loses edges as
contrast drops (not measured separately). `dct` ignores both, but not
mirroring: flipping an image left to right changes the sign of every
coefficient with an odd horizontal frequency. At half weight it gives the
best overall result (98% of copies have their original among
their 8 nearest images, against 96%).

### A bug the first run found

The first version gave a zero vector to any layout varying by less than 3
levels, as "flat". 23 textures (studded, polka-dotted, zigzagged) qualified:
their fine patterns average out at 32x32. But a zero vector is at squared
distance 1 from every image, while unrelated images are about 2 apart, so
these 23 became everyone's nearest image. With `phash` alone one of them
was among the 8 nearest of 921 images, 1,800 images were nobody's
neighbor, and the label scores fell below chance (2.7% same class). Now only
a constant image counts as flat (`LAYOUT_FLAT`). With `phash` alone the
most-cited image now has 25 citations and 5 images are nobody's neighbor.

A mirror-proof `dct` came next; see below.

To run, after `make_mix.py` and `cache.py` as below (about 20 s each):

```
F=benchmarks/features
BASE="rgb+rgb@bands3+edges@bands3+lbp@bands3"
python -I $F/scores.py "shipped=$BASE" \
    "+freq=$BASE+freq@bands3" "freq for lbp=rgb+rgb@bands3+edges@bands3+freq@bands3" \
    "+dct=$BASE+dct" "+dct*0.5=$BASE+dct*0.5" "+phash=$BASE+phash" \
    "freq alone=freq@bands3" "dct alone=dct" "phash alone=phash"
python -I $F/duplicates.py $F/data/mix "shipped=$BASE" "+freq=$BASE+freq@bands3" \
    "+dct=$BASE+dct" "+dct*0.5=$BASE+dct*0.5" "+phash=$BASE+phash" "dct alone=dct"
```

The thresholds were set by hand.

## Mirror-proof layout (in groupImg, not judged by eye yet)

`dct` found brighter copies but lost mirrored ones: mirroring an image left
to right flips the sign of the 32 of its 63 coefficients that have an odd
horizontal frequency. Two blocks undo that:

- **`dctm`** takes those 32 as absolute values. A mirrored copy gets the
  same vector, give or take rounding in the 32x32 squash: at most 0.18 of
  the 0-4 range, in images with almost no layout. The block forgets how
  the signs of those 32 relate to each other, so some arrangements that
  are not mirror images of each other look the same too.
- **`dctc`** flips the images whose left half is darker than their right
  (by the broadest left-right coefficient), so all of them face the same
  way. It keeps every sign, but an image about as bright on the left as on
  the right can fall either way: 2 of the 2,000 test thumbnails ended up
  more than 0.1 from their own mirror image, one 1.24.

| option | same class | same kind | purity k=12, 4 seeds | nearest is own: cropped / brighter / mirrored / all | same group k=50: brighter / mirrored / all |
|---|---|---|---|---|---|
| before | 27.6% | 81.3% | 20.6% | 100 / **67** / 100 / 93% | **57** / 100 / 90% |
| + dct at half weight | 26.5% | 82.4% | | 100 / 92 / **82** / 95% | 81 / **90** / 92% |
| **+ dctm at half weight (shipped)** | **27.5%** | **83.4%** | **22.3%** | 100 / 88 / 100 / 97% | 70 / 100 / 92% |
| + dctm | 25.8% | 82.6% | 22.1% | 98 / 95 / 100 / 98% | 81 / 100 / 94% |
| + dctc at half weight | 26.9% | 82.9% | 21.9% | 98 / 92 / 100 / 98% | 73 / 100 / 93% |
| + dctc | 24.8% | 81.8% | | 95 / 95 / 100 / 98% | 84 / 100 / 94% |

`dctm` at half weight (1/9 of the distance) is the only option that loses
nothing on the label scores while finding every mirrored copy and more of
the brighter ones (88% against 67%). Its group purity at k = 12 was higher
with each of 4 seeds; at k = 25 and 50 the differences stay within the
point or so that seeds change. At full weight it finds more brighter copies
(95%) but loses 1.8 points of same class. `dctc` does no better and has the
edge case above. Adding `freq` on top changes little (27.4% same class,
98% of copies found).

### What changed in groupImg

- **`layout()`** in `cli/groupimg.py` and `gui/groupImgGUI.py` computes
  `dctm`, and `WEIGHTS` gives it 1/9 of the distance and the four other
  parts 2/9 each. `describe()` gives the config
  `rgb+rgb@bands3+edges@bands3+lbp@bands3+dctm*0.5` to within 2e-8.
- 376 values per image instead of 313 (1.5 KB instead of 1.25 KB).
- The descriptor takes 0.80 ms per thumbnail instead of 0.71. The CLI
  groups the 2,000 test images (k = 12) in 2.55 s instead of 2.21 s, the
  median of 6 runs each.

### Next

- Judge it by eye: run `groupImg.py` before and after, or a blind round
  (`judge.py ... "before=$BASE" "after=$BASE+dctm*0.5"`).
- A blind round with `freq`.
- Making the histograms hold up to brightness and contrast changes, if a
  brighter copy should still count as the same image.

To run, as above:

```
python -I $F/scores.py "before=$BASE" "+dctm*0.5=$BASE+dctm*0.5" "+dctm=$BASE+dctm" \
    "+dctc*0.5=$BASE+dctc*0.5" "+dctc=$BASE+dctc"
python -I $F/duplicates.py $F/data/mix "before=$BASE" "+dctm*0.5=$BASE+dctm*0.5" \
    "+dctm=$BASE+dctm" "+dctc*0.5=$BASE+dctc*0.5" "+dctc=$BASE+dctc"
```

## Files

- `descriptors.py`: every block tried, as configs such as
  `rgb+rgb@bands3+edges@bands3+lbp@bands3` (see its docstring).
  `describe()` in the CLI gives exactly what
  `rgb+rgb@bands3+edges@bands3+lbp@bands3+dctm*0.5` gives.
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
