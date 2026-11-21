# e-means vs groupImg's k-means

An investigation into whether [e-means](https://arxiv.org/abs/2611.09501)
(Meta FAIR, 2026, shipped in FAISS 1.16 as `faiss.contrib.e_means`) is worth
adopting in groupImg.

**Conclusion: not worth switching.** e-means didn't produce better groupings
here, and the clustering step isn't groupImg's bottleneck. A cheaper change
(squared-Euclidean distance instead of Manhattan) gets most of the speed
without a new dependency.

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

## Files

- `bench.py`: loads the `K_means` class from `cli/groupimg.py` and compares it
  against a NumPy e-means written from the rules in the FAISS README (faiss
  could not be installed in the environment used). Data is synthetic 64-bin
  RGB histograms in percent, the same feature `read_image` builds, drawn from
  `k` true groups.
- `control.py`: plain Lloyd k-means with squared-L2 distance, k-means++ seeding
  and 10 restarts, on the same data. It separates "e-means" from "switching
  away from Manhattan distance".
- `io_bench.py`: time to load one 12-megapixel JPEG and build its histogram.
- `results_*.txt`: output of the runs below (4-core Linux container).

Run from the repo root (needs numpy and Pillow):

```
python benchmarks/e-means/bench.py cli/groupimg.py
python benchmarks/e-means/control.py cli/groupimg.py
python benchmarks/e-means/io_bench.py /some/tmp/dir
```

ARI measures agreement with the true groups (1.0 = perfect).

## Results (10 restarts each)

| images | k | current groupImg | Lloyd squared-L2 | e-means |
|---|---|---|---|---|
| 5,000 | 20 | 6.5 s, ARI 0.945 | 0.6 s, 0.938 | 0.3 s, 1.000 |
| 50,000 | 20 | 77 s, 0.937 | 11.6 s, 0.999 | 3.2 s, 0.999 |
| 200,000 | 5 | 88 s, 1.000 | 47 s, 1.000 | 12.7 s, 1.000 |
| 200,000 | 20 | 371 s, 1.000 | 77 s, 0.935 | 13.8 s, 0.935 |

Feature extraction: about 98 ms per 12 MP JPEG (`io_bench.py`).

## Takeaways

- **Quality: no consistent winner.** At k=20 every method sometimes gets 1.0
  and sometimes about 0.94. That's seeding and local minima, not the algorithm.
  A single e-means run isn't safe (ARI 0.71 at n=5,000, k=5), so restarts are
  still needed.
- **Speed: real but irrelevant.** Loading 200k photos takes over an hour.
  Clustering them takes about 6 minutes with the current code. Typical folders
  cluster in seconds.
- **Most of the speedup comes from the distance.** The current code assigns by
  Manhattan distance but updates centres with the mean, which minimizes squared
  L2. The two are mismatched, and Manhattan distance can't use matrix
  multiplication. Lloyd with squared L2 is already 5-6x faster on large inputs.
- **Cost of adopting.** faiss would become a dependency of a project that
  currently needs only numpy and Pillow. Alternatively the project would carry
  its own copy of a brand-new algorithm.

## Caveats

- The e-means here is a reimplementation from the FAISS README, not the
  reference code. The initial `N_j`/`S_j` values were a guess (`B/k`).
- The data is synthetic. Real photo collections may behave differently.
- The paper itself (arxiv.org) could not be fetched from the environment used.

## Possible follow-up

Switch `distances`, `init_clusters` and the cost in `rearrange_clusters`
(in both `cli/groupimg.py` and `gui/groupImgGUI.py`) to squared Euclidean
distance computed via matrix multiplication.
