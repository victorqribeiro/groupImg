"""Compare groupImg versions and e-means on CIFAR-10 (real photos, 32x32).

Usage:
  python benchmarks/e-means/cifar.py CIFAR_DIR name=path/to/groupimg.py [name=...]
         [--sizes 2000,10000] [--groups 5,10,20]

--sizes is how many images to cluster (taken from the end, so 10000 is the
official test batch; at most 60000). Default: 10000.

CIFAR_DIR holds the files of the binary version of CIFAR-10
(https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz): data_batch_1..5.bin
and test_batch.bin. Each record is 1 label byte and 3072 pixel bytes (R, G, B
planes of 32x32). No pickle is involved.

Features are built with groupImg's own read_image, so they are exactly what the
CLI computes for these images. CIFAR classes (cat, truck, ...) are not colour
groups, so agreement with the labels is only a weak signal. The main measures
are the squared-L2 cost (what every method here tries to minimise), the L1 cost
(what the old code minimised) and how much two independent runs agree.
"""
import io, os, sys, time, types, random, argparse
import numpy as np
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("data_dir")
ap.add_argument("versions", nargs="+", help="name=path/to/groupimg.py")
ap.add_argument("--sizes", default="10000")
ap.add_argument("--groups", default="5,10,20")
args = ap.parse_args()
data_dir = args.data_dir
versions = dict(a.split("=", 1) for a in args.versions)
sizes = [int(v) for v in args.sizes.split(",")]
groups = [int(v) for v in args.groups.split(",")]

# e-means, ARI and cost helpers from bench.py, without running its benchmark
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=None))
ns = {}
src = open(os.path.join(here, "bench.py")).read()
helpers = src[src.index("def emeans"):src.index("rng = np.random.default_rng(0)")]
exec("import math, random\nimport numpy as np\n" + helpers, ns)
emeans, ari, costs = ns["emeans"], ns["ari"], ns["costs"]


def load_kmeans(path):
    g = {}
    exec(open(path).read().split("ap = argparse.ArgumentParser()")[0], g)
    g["pbar"] = types.SimpleNamespace(update=lambda n: None)
    return g["K_means"]


def load_cifar():
    cache = os.path.join(data_dir, "groupimg_features.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x"], z["y"]
    files = [f"data_batch_{i}.bin" for i in range(1, 6)] + ["test_batch.bin"]
    raw = np.concatenate([np.fromfile(os.path.join(data_dir, f), np.uint8).reshape(-1, 3073)
                          for f in files])
    y = raw[:, 0].astype(np.int64)
    imgs = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    km = load_kmeans(next(iter(versions.values())))(5, False, 128)
    x = []
    for a in imgs:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="PNG")
        buf.seek(0)
        v, _ = km.read_image(buf)
        x.append(v)
    x = np.array(x)
    np.savez(cache, x=x, y=y)
    return x, y


def run_groupimg(K, x, k, seed):
    random.seed(seed); np.random.seed(seed)
    km = K(k)
    km.data = x
    km.rearrange_clusters()
    return np.asarray(km.cluster)


def run_emeans(x, k, seed):
    best = None
    for s in range(10):
        _, lab = emeans(x, k, seed=seed * 100 + s)
        c = costs(x, lab, k)[1]
        if best is None or c < best[0]:
            best = (c, lab)
    return best[1]


x_all, y_all = load_cifar()
print(f"CIFAR-10: {len(x_all)} images, {x_all.shape[1]} features")
methods = {f"groupImg {name}": (lambda K: lambda x, k, s: run_groupimg(K, x, k, s))(load_kmeans(p))
           for name, p in versions.items()}
methods["e-means (10 runs)"] = run_emeans
seeds = [0, 1, 2]
print(f"{'n':>6} {'k':>3} {'method':<22} {'time s':>7} {'MSE/pt':>8} {'L1/pt':>7} "
      f"{'ARI lbl':>7} {'ARI run':>7} {'min grp':>7}")
for n in sizes:
    x, y = x_all[-n:], y_all[-n:]   # n=10000 is the official test batch
    for k in groups:
        for name, fn in methods.items():
            labs, times = [], []
            for s in seeds:
                t = time.time(); labs.append(fn(x, k, s)); times.append(time.time() - t)
            res = [(costs(x, l, k), l) for l in labs]
            (l1, l2), lab = min(res, key=lambda r: r[0][1])
            sizes = np.bincount(lab, minlength=k)
            agree = np.mean([ari(labs[i], labs[j]) for i in range(3) for j in range(i + 1, 3)])
            print(f"{n:>6} {k:>3} {name:<22} {np.mean(times):>7.2f} {l2:>8.2f} {l1:>7.2f} "
                  f"{ari(lab, y):>7.3f} {agree:>7.3f} {sizes.min():>7}", flush=True)
