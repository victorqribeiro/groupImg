"""Control: plain Lloyd k-means (squared L2, k-means++ seeding, 10 restarts)."""
import sys, time
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1]]
src = open(__file__.replace("control.py", "bench.py")).read().split("rng = np.random.default_rng(0)")[0]
exec(src)


def sqd(x, c):
    return np.maximum((x ** 2).sum(1)[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None], 0)


def lloyd(x, k, rng, iters=300):
    c = [x[rng.integers(len(x))]]
    d2 = sqd(x, np.array(c))[:, 0]
    while len(c) < k:
        c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, sqd(x, np.array(c[-1:]))[:, 0])
    c = np.array(c)
    lab = None
    for _ in range(iters):
        new = sqd(x, c).argmin(1)
        if lab is not None and (new == lab).all():
            break
        lab = new
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = x[m].mean(0)
    return lab


rng = np.random.default_rng(0)
for n in SIZES:
    for k in [5, 20]:
        x, truth = histograms(n, k, rng)   # same data stream as bench.py
        r = np.random.default_rng(0)
        t = time.time(); best = None
        for _ in range(10):
            lab = lloyd(x, k, r)
            l2 = costs(x, lab, k)[1]
            if best is None or l2 < best[0]:
                best = (l2, lab)
        tl = time.time() - t
        l1, l2 = costs(x, best[1], k)
        print(f"{n:>7} {k:>3} {'Lloyd L2 ++ (10 runs)':<22} {tl:>8.2f} {l1:>8.2f} {l2:>8.2f} {ari(best[1], truth):>6.3f}", flush=True)
