"""Label scores for descriptor configs, without building a judging page.

  python -I benchmarks/features/scores.py NAME=SPEC [NAME=SPEC ...]
         [--cache benchmarks/features/out/cache.npz] [--k 12 25 50] [--seed 1]

Prints, for each config, the scores in the README's tables: the share of
each image's 8 nearest images with the same class and the same kind (photo
or texture), and group purity for each k. Clustering is the CLI's k-means
with the same seed for every config, as in judge.py. Use it to shortlist
configs before a blind round.
"""
import os
import sys
import time
import random
import argparse
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import descriptors as d  # noqa: E402
from cache import read_cache  # noqa: E402
from judge import NEIGHBORS, load_kmeans, nearest, ari, ms_per_image  # noqa: E402

TIMING_SAMPLE = 300


def purity(cluster, labels, k):
    """Share of images whose class is the most common in their group."""
    majority = sum(np.bincount(np.unique(labels[cluster == c], return_inverse=True)[1]).max()
                   for c in range(k) if (cluster == c).any())
    return majority / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="NAME=SPEC")
    ap.add_argument("--cache", default=os.path.join(here, "out", "cache.npz"))
    ap.add_argument("--k", type=int, nargs="+", default=[12, 25, 50])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    configs = [c.split("=", 1) if "=" in c else (c, c) for c in args.configs]
    for _, spec in configs:
        d.parse(spec)  # fail early on a typo

    data = read_cache(args.cache)
    thumbs, osizes, labels = data["thumbs"], data["osize"], data["labels"]
    coarse = np.array([l.split(os.sep)[0] for l in labels])
    n = len(thumbs)
    timing = random.Random(args.seed).sample(range(n), min(TIMING_SAMPLE, n))
    K = load_kmeans()

    t = time.perf_counter()
    counts = d.count_all(thumbs, osizes, d.keys_of([s for _, s in configs]))
    print(f"blocks for {n} images: {time.perf_counter() - t:.1f} s\n")

    ks = " | ".join(f"purity k={k}" for k in args.k)
    print(f"| option | values | ms/image | same class | same kind | {ks} | ARI k={args.k[0]} |")
    print("|---" * (6 + len(args.k)) + "|")
    for name, spec in configs:
        x = d.combine(counts, spec)
        nn = nearest(x, range(n), NEIGHBORS)
        row = [name, str(x.shape[1]),
               f"{ms_per_image([thumbs[i] for i in timing], osizes[timing], spec):.2f}",
               f"{(labels[nn] == labels[:, None]).mean():.1%}",
               f"{(coarse[nn] == coarse[:, None]).mean():.1%}"]
        first = None
        for k in args.k:
            random.seed(args.seed)
            np.random.seed(args.seed)
            km = K(k)
            km.data = x  # as judge.py does, so the numbers match its rounds
            km.rearrange_clusters()
            cluster = np.asarray(km.cluster)
            first = cluster if first is None else first
            row.append(f"{purity(cluster, labels, k):.1%}")
        row.append(f"{ari(first, labels):.3f}")
        print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
