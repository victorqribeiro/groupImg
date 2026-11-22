"""Near-duplicate test: does a copy of an image find its original?

  python -I benchmarks/features/duplicates.py FOLDER NAME=SPEC [NAME=SPEC ...]
         [--cache benchmarks/features/out/cache.npz] [--queries 200]
         [--k 12 50] [--seed 1]

FOLDER is the folder the cache was built from (cache.py FOLDER). The label
scores ask whether images of the same class end up together; this asks the
question groupImg was first written for, a folder of recovered files full of
resized, re-saved and cropped copies of the same photos. --queries images
of the cache are copied in each of these ways, in memory, and read back as
the CLI reads files:

  smaller    half the size, saved as a JPEG at quality 50
  thumbnail  at most 160 px, quality 75, like the previews inside EXIF data
  cropped    10% cut from every side, quality 90
  brighter   25% brighter and 15% less contrast, quality 90
  mirrored   flipped left to right, quality 90

Prints, per config and kind of copy:

  nearest    share of copies whose nearest image among all the originals
             is their own original
  top 8      the same, among the 8 nearest
  group      share of copies put in the same group as their original when
             the CLI's k-means splits originals and copies together, for
             each --k. "chance" is what random groups of the same sizes give.
"""
import io
import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
import descriptors as d  # noqa: E402
from cache import read_cache  # noqa: E402
from judge import load_kmeans, sqdist  # noqa: E402


def jpeg(img, quality):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf


def smaller(img):
    return jpeg(img.resize((max(img.width // 2, 1), max(img.height // 2, 1)), Image.LANCZOS), 50)


def thumbnail(img):
    img = img.copy()
    img.thumbnail((160, 160))
    return jpeg(img, 75)


def cropped(img):
    dx, dy = img.width // 10, img.height // 10
    return jpeg(img.crop((dx, dy, img.width - dx, img.height - dy)), 90)


def brighter(img):
    return jpeg(ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.25)).enhance(0.85), 90)


def mirrored(img):
    return jpeg(img.transpose(Image.FLIP_LEFT_RIGHT), 90)


COPIES = [smaller, thumbnail, cropped, brighter, mirrored]


def make_copies(root, paths):
    """Thumbnails and sizes of every copy of every image, as the CLI reads
    them, and the kind of copy (index in COPIES) of each."""
    thumbs, osizes, kinds = [], [], []
    for path in paths:
        img = ImageOps.exif_transpose(Image.open(os.path.join(root, path))).convert("RGB")
        for kind, fn in enumerate(COPIES):
            pixels, osize = d.load_thumbnail(fn(img))
            thumbs.append(pixels)
            osizes.append(osize)
            kinds.append(kind)
    return thumbs, np.array(osizes), np.array(kinds)


def table(title, columns, rows):
    print(f"\n{title}\n")
    print("| option | " + " | ".join(columns) + " |")
    print("|---" * (len(columns) + 1) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="the folder the cache was built from")
    ap.add_argument("configs", nargs="+", help="NAME=SPEC")
    ap.add_argument("--cache", default=os.path.join(here, "out", "cache.npz"))
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, nargs="+", default=[12, 50])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    configs = [c.split("=", 1) if "=" in c else (c, c) for c in args.configs]
    for _, spec in configs:
        d.parse(spec)  # fail early on a typo

    data = read_cache(args.cache)
    thumbs, osizes, paths = data["thumbs"], data["osize"], data["paths"]
    n = len(thumbs)
    queries = np.array(random.Random(args.seed).sample(range(n), min(args.queries, n)))
    cthumbs, cosizes, kinds = make_copies(os.path.abspath(args.folder), paths[queries])
    # the original of every copy
    source = np.repeat(queries, len(COPIES))
    print(f"{n} originals, {len(cthumbs)} copies of {len(queries)} of them")

    keys = d.keys_of([s for _, s in configs])
    counts = d.count_all(thumbs, osizes, keys)
    ccounts = d.count_all(cthumbs, cosizes, keys)
    K = load_kmeans()
    names = [fn.__name__ for fn in COPIES] + ["all"]

    def by_kind(hit):
        return [f"{hit[kinds == c].mean():.0%}" for c in range(len(COPIES))] + [f"{hit.mean():.0%}"]

    near, top8, groups = [], [], {k: [] for k in args.k}
    for name, spec in configs:
        x, y = d.combine(counts, spec), d.combine(ccounts, spec)
        dist = sqdist(y, x)
        own = dist[np.arange(len(y)), source]
        # how many originals are closer to the copy than its own original
        rank = (dist < own[:, None]).sum(axis=1)
        near.append([name] + by_kind(rank == 0))
        top8.append([name] + by_kind(rank < 8))
        both = np.concatenate([x, y])
        for k in args.k:
            random.seed(args.seed)
            np.random.seed(args.seed)
            km = K(k)
            km.data = both
            km.rearrange_clusters()
            cluster = np.asarray(km.cluster)
            share = np.bincount(cluster, minlength=k) / len(cluster)
            hit = cluster[n:] == cluster[source]
            groups[k].append([name] + by_kind(hit) + [f"{(share ** 2).sum():.0%}"])
        print(f"{name}: done", flush=True)

    table("Nearest original is its own", names, near)
    table("Own original among the 8 nearest", names, top8)
    for k in args.k:
        table(f"In the same group as its original, k = {k}", names + ["chance"], groups[k])


if __name__ == "__main__":
    main()
