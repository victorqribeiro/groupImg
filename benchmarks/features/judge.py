"""Build a blind comparison page for one round of descriptor configs.

  python -I benchmarks/features/judge.py --round 1 NAME=SPEC [NAME=SPEC ...]
         [--cache benchmarks/features/out/cache.npz] [--k 12] [--queries 30]
         [--seed 1]

SPEC is a config from descriptors.py, e.g. "rgb+rgb@bands3". NAME is only
shown after the reveal. Writes out/roundN/index.html; open it through
serve.py so votes are saved to results/roundN.json.

Every config gets the same query images, and its clustering gets the same
k and random seeds, so the descriptor is the only thing that differs.
"""
import os
import sys
import json
import time
import random
import argparse
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import descriptors as d  # noqa: E402
from cache import read_cache  # noqa: E402

NEIGHBORS = 8      # images per row on the neighbors page
NEAR, FAR = 12, 4  # images shown per group: closest to its center, farthest
TIMING_SAMPLE = 300


def load_kmeans():
    # the class part of the CLI, without its argument parsing and file moves
    src = open(os.path.join(here, "..", "..", "cli", "groupimg.py")).read()
    g = {}
    exec(src.split("ap = argparse.ArgumentParser()")[0], g)
    return g["K_means"]


def sqdist(a, b):
    dist = (a * a).sum(1)[:, None] - 2 * a @ b.T + (b * b).sum(1)[None, :]
    return np.maximum(dist, 0)


def nearest(x, idx, m):
    """The m nearest other images of each image in idx, closest first."""
    out = []
    for s in range(0, len(idx), 1000):
        q = np.asarray(idx[s:s + 1000])
        dist = sqdist(x[q], x)
        dist[np.arange(len(q)), q] = np.inf
        part = np.argpartition(dist, m, axis=1)[:, :m]
        order = np.take_along_axis(dist, part, 1).argsort(1)
        out.append(np.take_along_axis(part, order, 1))
    return np.concatenate(out)


def ari(a, b):
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda v: (v * (v - 1) / 2).sum()
    both, sa, sb = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = sa * sb / pairs(np.array([len(a)], float))
    return float((both - expected) / ((sa + sb) / 2 - expected))


def groups_of(x, cluster, k):
    groups = []
    for c in range(k):
        members = np.flatnonzero(cluster == c)
        if not len(members):
            continue
        dist = ((x[members] - x[members].mean(0)) ** 2).sum(1)
        order = members[dist.argsort()]
        groups.append({"size": len(members),
                       "near": order[:NEAR].tolist(),
                       "far": order[max(NEAR, len(order) - FAR):].tolist()})
    return sorted(groups, key=lambda g: -g["size"])


def ms_per_image(thumbs, osizes, spec):
    t = time.perf_counter()
    for img, osize in zip(thumbs, osizes):
        d.features(img, osize, spec)
    return (time.perf_counter() - t) / len(thumbs) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", help="NAME=SPEC")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--cache", default=os.path.join(here, "out", "cache.npz"))
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pick", choices=("random", "disagree"), default="random",
                    help="query images: at random, or among the tenth of images "
                         "whose nearest neighbors differ most between the configs")
    args = ap.parse_args()
    configs = [c.split("=", 1) if "=" in c else (c, c) for c in args.configs]
    for _, spec in configs:
        d.parse(spec)  # fail early on a typo

    data = read_cache(args.cache)
    thumbs, osizes = data["thumbs"], data["osize"]
    labels = data["labels"]
    coarse = np.array([l.split(os.sep)[0] for l in labels])
    n = len(thumbs)
    rng = random.Random(args.seed)
    queries = rng.sample(range(n), args.queries)
    timing = rng.sample(range(n), min(TIMING_SAMPLE, n))
    K = load_kmeans()

    t = time.perf_counter()
    counts = d.count_all(thumbs, osizes, d.keys_of([s for _, s in configs]))
    print(f"blocks for {n} images: {time.perf_counter() - t:.1f} s")

    results = []
    for name, spec in configs:
        x = d.combine(counts, spec)
        nn = nearest(x, range(n), NEIGHBORS)
        random.seed(args.seed)
        np.random.seed(args.seed)
        km = K(args.k)
        km.data = x
        t = time.perf_counter()
        km.rearrange_clusters()
        kmeans_s = time.perf_counter() - t
        cluster = np.asarray(km.cluster)
        majority = sum(np.bincount(np.unique(labels[cluster == c], return_inverse=True)[1]).max()
                       for c in range(args.k) if (cluster == c).any())
        results.append({
            "name": name, "spec": spec, "dims": x.shape[1],
            "ms": ms_per_image([thumbs[i] for i in timing], osizes[timing], spec),
            "kmeans_s": kmeans_s,
            "p8": float((labels[nn] == labels[:, None]).mean()),
            "p8_coarse": float((coarse[nn] == coarse[:, None]).mean()),
            "purity": float(majority / n),
            "ari": ari(cluster, labels),
            "nn": nn,
            "groups": groups_of(x, cluster, args.k),
        })
        print(f"{name}: {x.shape[1]} values, k-means {kmeans_s:.1f} s")

    if args.pick == "disagree":
        # where the configs agree the rows look the same and the question is
        # wasted, so ask only where they share the fewest neighbors
        shared = np.zeros(n)
        for a in range(len(results)):
            for b in range(a + 1, len(results)):
                shared += [len(set(p) & set(q)) for p, q in zip(results[a]["nn"], results[b]["nn"])]
        least = np.argsort(shared, kind="stable")[:n // 10].tolist()
        queries = random.Random(args.seed).sample(least, args.queries)

    page = {
        "round": args.round, "k": args.k, "n": n, "pick": args.pick,
        "sizes": data["tsize"].tolist(),
        "configs": [{k: r[k] for k in ("name", "spec", "dims", "ms", "kmeans_s",
                                       "p8", "p8_coarse", "purity", "ari")}
                    for r in results],
        "queries": [],
        "options": [],
    }
    for q in queries:
        order = list(range(len(results)))
        rng.shuffle(order)
        page["queries"].append({"image": q, "rows": [
            {"config": c, "images": results[c]["nn"][q].tolist()} for c in order]})
    order = list(range(len(results)))
    rng.shuffle(order)
    page["options"] = [{"config": c, "groups": results[c]["groups"]} for c in order]

    out_dir = os.path.join(os.path.dirname(args.cache), f"round{args.round}")
    os.makedirs(out_dir, exist_ok=True)
    html = open(os.path.join(here, "page.html")).read()
    blob = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html.replace("__ROUND__", str(args.round)).replace("__DATA__", blob))
    print("wrote", os.path.join(out_dir, "index.html"))


if __name__ == "__main__":
    main()
