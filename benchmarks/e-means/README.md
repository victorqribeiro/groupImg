# e-means vs groupImg's k-means

An investigation into whether [e-means](https://arxiv.org/abs/2611.09501)
(Meta FAIR, 2026, shipped in FAISS 1.16 as `faiss.contrib.e_means`) is worth
adopting in groupImg, and what was changed instead.

**Conclusion: e-means is not worth adopting.** groupImg now uses
squared-Euclidean distance with greedy k-means++ seeding. On real photos
(CIFAR-10) that makes clustering 2-8x faster and the groups tighter. On
synthetic data with known groups it finds the right groups more reliably than
the old code. It needs no new dependency.

## What e-means is

It minimizes the same squared-Euclidean loss as k-means, with mini-batch
updates built on exponential moving averages:

```
N_j <- (1 - alpha) * N_j + alpha * n_j   # running count of points in cluster j
S_j <- (1 - alpha) * S_j + alpha * s_j   # running sum of those points
c_j  = S_j / N_j                         # centroid
```

Batch size and starting step size come from fitted scaling laws. The step size
follows a cosine decay down to alpha0/10. Starved clusters are revived by
splitting busy ones, and the final centroids are exact means from the last
pass. The claim is about 10x fewer epochs than Lloyd's k-means at equal or lower
loss, measured mostly on large embedding datasets.

## What changed in groupImg

In `cli/groupimg.py` and `gui/groupImgGUI.py`:

- **Squared-Euclidean distance instead of Manhattan.** The old code assigned
  images by Manhattan (L1) distance but moved centres to the mean, which
  minimizes squared L2. The two disagreed. Squared L2 also runs as one matrix
  multiplication (`|x|^2 - 2 x.m + |m|^2`).
- **A tie tolerance that scales with the data.** The matrix form loses
  precision in proportion to the size of the vectors. With `-s`, image sizes
  in the thousands go into the features, and a fixed `1e-9` would be smaller
  than the rounding error.
- **Greedy k-means++ seeding.** Each new starting centre is the best of
  `2 + ln k` candidates, as in scikit-learn. With plain k-means++ the new
  distance often left two true groups sharing a centre. See "Seeding" below.
- Removed the unused `euclidian_distance` method, which in fact computed the
  Manhattan distance.

## Files

- `bench.py`: loads the `K_means` class from a given `groupimg.py` and compares
  it with a NumPy e-means written from the rules in the FAISS README (faiss
  could not be installed in the environment first used). The data is synthetic
  64-bin RGB histograms in percent, the feature `read_image` builds, drawn from
  `k` true groups.
- `control.py`: plain Lloyd k-means with squared L2, plain k-means++ seeding
  and 10 restarts, on the same data.
- `seeds.py`: runs two versions of `groupimg.py` with 10 seeds each on one
  synthetic case (5,000 images, k=20), to tell a real difference from a lucky
  seed.
- `cifar.py`: the old code, the new code and e-means on
  [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), real 32x32 photos.
  Features come from groupImg's own `read_image`.
- `io_bench.py`: time to load one 12-megapixel JPEG and build its histogram.
- `results_*.txt`: output of the runs below, on a 16-core Linux laptop.
  `results_bench_old.txt` is `bench.py` run on the old code (master before
  this change). The other `bench.py`/`seeds.py`/`cifar.py` results compare
  against it.

Run from the repo root (needs numpy and Pillow). `OLD.py` is the previous
version, e.g. `git show master:cli/groupimg.py > /tmp/old.py`:

```
python benchmarks/e-means/bench.py cli/groupimg.py
python benchmarks/e-means/control.py cli/groupimg.py
python benchmarks/e-means/seeds.py OLD.py cli/groupimg.py
python benchmarks/e-means/cifar.py CIFAR_DIR old=OLD.py new=cli/groupimg.py
python benchmarks/e-means/io_bench.py /some/tmp/dir
```

`CIFAR_DIR` is the extracted `cifar-10-batches-bin` folder from the binary
version (`cifar-10-binary.tar.gz`, md5 `c32a1d4ab5d03f1284b67883e8d87530`).
`cifar.py` caches features in that folder. `--sizes` picks how many images
(default 10000, the official test batch; up to 60000) and `--groups` picks
the k values.

Runs are kept to 10,000 images or fewer so the whole set takes a few minutes.

## Results

ARI measures agreement between two groupings (1.0 = identical). "ARI vs
truth" compares with the known groups, "ARI between runs" compares
independent runs with different seeds.

### CIFAR-10, 10,000 photos (`results_cifar.txt`)

Best of 3 seeds. Each groupImg seed does 10 restarts, and e-means also gets
10 runs per seed.

| k | method | time per seed | MSE/pt | L1/pt | ARI between runs | smallest group |
|---|---|---|---|---|---|---|
| 5 | old | 2.7 s | 877 | 76.7 | 0.990 | 786 |
| 5 | **new** | 1.3 s | **834** | 79.3 | 0.999 | 1531 |
| 5 | e-means | 0.3 s | 834 | 79.3 | 0.976 | 1514 |
| 10 | old | 9.6 s | 676 | 69.0 | 0.973 | 411 |
| 10 | **new** | 1.6 s | **642** | 69.3 | 0.978 | 554 |
| 10 | e-means | 0.3 s | 643 | 69.3 | 0.962 | 541 |
| 20 | old | 26.6 s | 539 | 62.5 | 0.790 | 112 |
| 20 | **new** | 3.4 s | **504** | 63.2 | 0.697 | 90 |
| 20 | e-means | 0.3 s | 506 | 63.2 | 0.721 | 122 |

An earlier run on all 60,000 images showed the same picture (k=20: old 313 s,
new 39 s, e-means 3.4 s; MSE 538 / 499 / 500).

CIFAR's labels (airplane, cat, ...) aren't colour groups. Every method scores
an ARI of about 0.03 against them, so the labels can't say which grouping is
better. What can be measured:

- **Tightness:** the new code's groups are 5-7% tighter by squared distance.
  They are 1-4% looser by L1, which is what the old code minimized. Each
  version wins on its own measure. Squared L2 is the one the mean actually
  minimizes.
- **Stability:** at k=5 and k=10, independent runs agree as much as or more
  than with the old code. At k=20 they agree less (0.70 vs 0.79). Real photos
  don't fall into 20 clean colour groups, so several groupings are about
  equally good there. e-means is no more stable.
- **Balance:** the smallest group is larger with the new code at k=5 and k=10
  (fewer near-empty folders), and slightly smaller at k=20.

### Synthetic data with known groups (`results_bench*.txt`, `results_control.txt`)

10 restarts each, one seed:

| images | k | old | new | plain Lloyd L2 | e-means (10 runs) |
|---|---|---|---|---|---|
| 2,000 | 20 | 0.94 s, ARI 1.000 | 0.20 s, 1.000 | 0.10 s, 0.937 | 0.09 s, 1.000 |
| 5,000 | 20 | 3.63 s, 0.939 | 0.60 s, 0.999 | 0.39 s, 0.932 | 0.21 s, 0.999 |
| 10,000 | 5 | 1.03 s, 1.000 | 0.73 s, 1.000 | 0.24 s, 1.000 | 0.31 s, 1.000 |
| 10,000 | 20 | 7.27 s, 0.935 | 1.23 s, 1.000 | 1.50 s, 0.871 | 0.35 s, 1.000 |

### Seeding (`results_seeds.txt`)

5,000 synthetic images, k=20, 10 seeds of 10 restarts each:

| version | time per seed | seeds that found the true groups |
|---|---|---|
| old (Manhattan, plain k-means++) | 4.0 s | 5/10 |
| squared L2, plain k-means++ | 0.5 s | 0/10 |
| new (squared L2, greedy k-means++) | 0.6 s | 10/10 |

Switching the distance alone made this case worse. The plain squared-L2
version is not in the repo; it was the first draft of this change. The cause
was the seeding, not the distance. A second synthetic set with other settings
showed the same pattern (k=20: 1/10 → 10/10, k=10: 8/10 → 10/10). Seeding with
the old Manhattan weights helped only a little (2/10).

### Feature extraction

About 67 ms per 12-megapixel JPEG on this machine (`results_io.txt`; it was
98 ms on the 4-core container first used). The test JPEG is random noise,
which decodes more slowly than a real photo.

## Takeaways

- **e-means is the fastest, but the difference is irrelevant.** Clustering
  10,000 photos takes 3.4 s with the new code against 0.3 s with e-means.
  Reading those photos takes about 11 minutes on one core. The CLI reads in
  parallel, which cuts that, but it is still minutes. e-means' groupings are no
  better or more stable than the new code's.
- **e-means' cost.** faiss would become a dependency of a project that needs
  only numpy and Pillow. Alternatively the project would carry its own copy of
  a brand-new algorithm. A single e-means run isn't safe (ARI 0.71 at
  n=2,000, k=5), so restarts are still needed.
- **Squared L2 with greedy seeding is the better default.** It is 2-8x faster
  than the old code on CIFAR-10 (1.4-6x on the synthetic sets) and tighter by
  the measure the mean minimizes. On
  synthetic data it finds the true groups more reliably.

## Caveats

- The e-means here is a reimplementation from the FAISS README, not the
  reference code. The initial `N_j`/`S_j` values were a guess (`B/k`).
- CIFAR-10 images are tiny (32x32) and its labels aren't colour groups.
  Whether a grouping "looks right" to a person wasn't measured.
- Timings include numpy's multithreaded matrix multiply, which helps the new
  code and e-means but not the old Manhattan code.
- `-s` still puts raw pixel sizes (thousands) next to percentages (0-100), so
  image size outweighs colour. Squared distance makes that stronger. That
  predates this change and is left as is.
