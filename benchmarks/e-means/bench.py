"""Compare groupImg's current K_means against e-means on groupImg-style features.

e-means is re-implemented here from the rules documented in
faiss/contrib/e_means/README.md (EMA count/sum updates, scaling laws (1)-(3),
cosine step decay to alpha0/10, cluster revival, exact means after last pass).
"""
import sys, time, math, random
import numpy as np

# --- load the K_means class from the repo without running its CLI part ---
src = open(sys.argv[1]).read().split("ap = argparse.ArgumentParser()")[0]
import types; sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=None)); ns = {}
exec(src, ns)
K_means = ns["K_means"]


def emeans(x, k, nepochs=5, seed=1234):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    law = 0.00439175 if d <= 16 else 0.035134
    B = law * k ** 0.134749 * n ** 0.876473 * nepochs ** 0.813344
    B = int(min(2 ** round(math.log2(max(B, 1))), max(n // 2, 1)))
    a0 = 0.8244 * k ** -0.4617 * B ** 0.9003 * n ** -0.4498 * nepochs ** -0.1057
    a0 = min(max(a0, 1e-4), 1.0)
    steps_per_pass = math.ceil(n / B)
    total = int(math.ceil(nepochs * steps_per_pass))
    c = x[rng.choice(n, k, replace=False)].astype(np.float64)
    N = np.full(k, B / k)
    S = c * N[:, None]
    labels = np.empty(n, dtype=np.int64)
    step = 0
    for ep in range(math.ceil(nepochs)):
        perm = rng.permutation(n)
        last = ep == math.ceil(nepochs) - 1
        for s in range(steps_per_pass):
            if step >= total:
                break
            idx = perm[s * B:(s + 1) * B]
            xb = x[idx]
            dist = (xb ** 2).sum(1)[:, None] - 2 * xb @ c.T + (c ** 2).sum(1)[None]
            lab = dist.argmin(1)
            labels[idx] = lab
            nj = np.bincount(lab, minlength=k).astype(np.float64)
            sj = np.zeros((k, d))
            np.add.at(sj, lab, xb)
            a = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            N = (1 - a) * N + a * nj
            S = (1 - a) * S + a * sj
            # revival: split the busiest cluster into an under-used one
            for j in np.where(N < 0.1 * N.mean())[0]:
                big = N.argmax()
                eps = 1e-3 * (rng.standard_normal(d))
                c_big = S[big] / N[big]
                N[big] /= 2; S[big] = c_big * N[big]
                N[j] = N[big]; S[j] = (c_big + eps) * N[j]
            c = S / N[:, None]
            step += 1
    # exact means from the last pass's assignments
    for j in range(k):
        m = labels == j
        if m.any():
            c[j] = x[m].mean(0)
    return c, labels


def histograms(n, groups, rng):
    """64-bin colour histograms (percentages), like K_means.read_image makes."""
    bases = rng.dirichlet(np.full(64, 0.3), size=groups)
    truth = rng.integers(0, groups, n)
    conc = rng.uniform(20, 200, n)[:, None]
    x = np.array([rng.dirichlet(bases[t] * c + 1e-3) for t, c in zip(truth, conc)]) * 100
    return x, truth


def ari(a, b):
    from math import comb
    ct = np.zeros((a.max() + 1, b.max() + 1), dtype=np.int64)
    np.add.at(ct, (a, b), 1)
    s = sum(comb(int(v), 2) for v in ct.ravel())
    sa = sum(comb(int(v), 2) for v in ct.sum(1))
    sb = sum(comb(int(v), 2) for v in ct.sum(0))
    e = sa * sb / comb(len(a), 2)
    return (s - e) / (0.5 * (sa + sb) - e)


def costs(x, lab, k):
    c = np.array([x[lab == j].mean(0) if (lab == j).any() else x.mean(0) for j in range(k)])
    r = x - c[lab]
    return np.abs(r).sum() / len(x), (r ** 2).sum() / len(x)


# kept small so a full run takes a few minutes
SIZES = [500, 2000, 5000, 10000]

rng = np.random.default_rng(0)
print(f"{'n':>7} {'k':>3} {'algo':<22} {'time s':>8} {'L1/pt':>8} {'MSE/pt':>8} {'ARI':>6}")
for n in SIZES:
    for k in [5, 20]:
        x, truth = histograms(n, k, rng)
        random.seed(0); np.random.seed(0)
        km = K_means(k)
        km.data = x
        t = time.time(); km.rearrange_clusters(); tk = time.time() - t
        l1, l2 = costs(x, km.cluster, k)
        print(f"{n:>7} {k:>3} {'groupImg (10 restarts)':<22} {tk:>8.2f} {l1:>8.2f} {l2:>8.2f} {ari(km.cluster, truth):>6.3f}")
        t = time.time(); _, lab = emeans(x, k); te = time.time() - t
        l1, l2 = costs(x, lab, k)
        print(f"{n:>7} {k:>3} {'e-means (1 run, 5 ep)':<22} {te:>8.2f} {l1:>8.2f} {l2:>8.2f} {ari(lab, truth):>6.3f}")
        best = None; t = time.time()
        for s in range(10):
            _, lab = emeans(x, k, seed=s)
            l2s = costs(x, lab, k)[1]
            if best is None or l2s < best[0]:
                best = (l2s, lab)
        te = time.time() - t
        l1, l2 = costs(x, best[1], k)
        print(f"{n:>7} {k:>3} {'e-means (10 runs)':<22} {te:>8.2f} {l1:>8.2f} {l2:>8.2f} {ari(best[1], truth):>6.3f}")
        sys.stdout.flush()
