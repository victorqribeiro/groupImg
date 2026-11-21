"""Decode a folder of images once, for every descriptor experiment.

  python -I benchmarks/features/cache.py FOLDER [--out benchmarks/features/out]

Walks FOLDER recursively. Each image's label is its folder relative to
FOLDER ("photos/church"), used only for secondary scores. Writes:

  OUT/cache.npz    128 px thumbnails as the CLI will read them (EXIF-rotated,
                   transparency on white), original sizes, paths, labels.
                   Stored as one flat pixel buffer plus offsets, so loading
                   needs no pickle.
  OUT/thumbs/I.jpg 256 px copies for the judging pages.
"""
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageOps

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from descriptors import load_thumbnail  # noqa: E402

EXTENSIONS = ('.jpg', '.jpeg', '.png')
FEATURE_SIZE = 128   # the CLI's default -r
DISPLAY_SIZE = 256


def find_images(root):
    found = []
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            if f.lower().endswith(EXTENSIONS):
                found.append(os.path.join(folder, f))
    return found


def load(path, thumbs_dir, i):
    try:
        pixels, osize = load_thumbnail(path, FEATURE_SIZE)
        if not thumbs_dir:
            return pixels, osize, pixels.shape[1::-1]
        show = ImageOps.exif_transpose(Image.open(path))
        show.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        show.convert('RGB').save(os.path.join(thumbs_dir, f"{i}.jpg"), quality=85)
        return pixels, osize, show.size
    except Exception as e:
        print("Error reading", path, e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--no-thumbs", action="store_true", help="skip the 256 px copies (timing runs)")
    args = ap.parse_args()
    root = os.path.abspath(args.folder)
    thumbs_dir = None if args.no_thumbs else os.path.join(args.out, "thumbs")
    os.makedirs(thumbs_dir or args.out, exist_ok=True)

    paths = find_images(root)
    t = time.perf_counter()
    with ThreadPoolExecutor(os.cpu_count()) as pool:
        results = list(pool.map(load, paths, [thumbs_dir] * len(paths), range(len(paths))))
    ok = [i for i, r in enumerate(results) if r is not None]
    if len(ok) != len(paths):
        # thumbnails are numbered by position, so renumber after a failure
        sys.exit("some images could not be read; remove them and run again")
    pixels = [r[0] for r in results]
    offsets = np.cumsum([0] + [p.size for p in pixels])
    rel = [os.path.relpath(p, root) for p in paths]
    np.savez(os.path.join(args.out, "cache.npz"),
             pixels=np.concatenate([p.ravel() for p in pixels]),
             offsets=offsets,
             shapes=np.array([p.shape[:2] for p in pixels], np.int32),
             osize=np.array([r[1] for r in results], np.int32),
             tsize=np.array([r[2] for r in results], np.int32),
             paths=np.array(rel),
             labels=np.array([os.path.dirname(p) for p in rel]))
    print(f"{len(paths)} images in {time.perf_counter() - t:.1f} s -> {args.out}")


def read_cache(path):
    """-> dict with thumbs (list of HxWx3 arrays) and the stored arrays."""
    z = np.load(path)
    data = {k: z[k] for k in z.files}
    px, off, shp = data.pop("pixels"), data["offsets"], data["shapes"]
    data["thumbs"] = [px[off[i]:off[i + 1]].reshape(shp[i][0], shp[i][1], 3)
                      for i in range(len(shp))]
    return data


if __name__ == "__main__":
    main()
