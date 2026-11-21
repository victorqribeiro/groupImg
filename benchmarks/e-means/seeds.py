"""Seed sweep on bench.py's synthetic data (5,000 images, k=20).

Usage: python benchmarks/e-means/seeds.py OLD_groupimg.py NEW_groupimg.py

Runs each version with 10 seeds (10 restarts each) and prints the cost and
ARI of every run, to tell a real difference from a lucky seed.
"""
import sys, time, random, types, numpy as np
sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=None))
import os
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")
src = open(here).read()
ns = {}; exec("import math, random\nimport numpy as np\n" + src[src.index("def emeans"):src.index("rng = np.random.default_rng(0)")], ns)
def load(p):
    g = {}; exec(open(p).read().split("ap = argparse.ArgumentParser()")[0], g); return g["K_means"]
Ks = {"old": load(sys.argv[1]), "new": load(sys.argv[2])}
rng = np.random.default_rng(0)
for n in [500, 5000]:          # same data stream as bench.py
    for k in [5, 20]:
        x, truth = ns["histograms"](n, k, rng)
        if k != 20 or n < 5000: continue
        for name, K in Ks.items():
            for s in range(10):
                random.seed(s); np.random.seed(s)
                km = K(k); km.data = x
                t = time.time(); km.rearrange_clusters(); dt = time.time() - t
                l1, l2 = ns["costs"](x, km.cluster, k)
                print(f"{n:>7} {k} {name} seed={s} {dt:7.2f}s MSE={l2:7.2f} ARI={ns['ari'](np.asarray(km.cluster), truth):.3f}", flush=True)
