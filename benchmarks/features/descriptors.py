"""Image descriptors for the groupImg feature benchmark.

A descriptor is a list of blocks. Each block is one histogram of the whole
image, or one per region of it:

  rgb     joint RGB histogram, 4 bins per channel (64). groupImg uses it now.
  hsv     52 bins: black, 3 grays, then 12 hues x 2 saturations x 2 values
  edges   gradient orientation: 8 unsigned directions plus 1 "flat" bin
  lbp     rotation-invariant uniform local binary patterns (10 bins)
  size    log area and log aspect ratio of the original image (2 values)

"@q2x2" computes a block per quadrant, "@bands3" per horizontal third.

A config joins blocks with "+", each with an optional weight:
"rgb*0.4+rgb@bands3*0.35+edges@bands3*0.25". Weights default to 1 and are
rescaled to sum to 1.

combine() divides every histogram by its pixel count and takes the square
root. Each histogram then has length 1, and the squared Euclidean distance
between two of them is twice their squared Hellinger distance (0 to 2). A
block's weight is its share of the distance, split evenly between its
regions. The size block is already on a 0-1 scale and is used as is.

The prefix "raw:" skips the square root and keeps histograms in percent,
exactly as groupImg does now.
"""
import math
import re
import numpy as np
from PIL import Image, ImageOps, ExifTags

BINS = {'rgb': 64, 'hsv': 52, 'edges': 9, 'lbp': 10, 'size': 2}
REGIONS = {None: 1, 'q2x2': 4, 'bands3': 3}
# a gradient below this (in 0-255 levels across 2 pixels) counts as flat
EDGE_THRESHOLD = 20.0
# a neighbor must be this much brighter than the center to set its LBP bit,
# so JPEG noise in flat areas doesn't make random patterns
LBP_THRESHOLD = 3.0
# neighbors of a 3x3 window in circular order, as offsets from its corner
LBP_RING = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

BLOCK = re.compile(r'(rgb|hsv|edges|lbp|size)(?:@(q2x2|bands3))?(?:\*([0-9.]+))?')


def load_thumbnail(path, size=128):
    """Open an image the way the CLI will: shrink it, apply the EXIF
    rotation, flatten transparency onto white. Returns the RGB array and the
    original size as displayed (width, height)."""
    img = Image.open(path)
    w, h = img.size
    # thumbnail() decodes JPEGs at reduced size, so this is cheap; the EXIF
    # data survives it, so rotating afterwards only moves a small image
    img.thumbnail((size, size))
    if img.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8):
        w, h = h, w
    img = ImageOps.exif_transpose(img)
    if img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info:
        img = img.convert('RGBA')
        white = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, img)
    return np.asarray(img.convert('RGB')), (w, h)


def parse(spec):
    """'raw:rgb+edges@q2x2*0.5' -> (raw, [(name, layout, weight), ...])"""
    raw = spec.startswith('raw:')
    blocks = []
    for part in spec[4 if raw else 0:].split('+'):
        m = BLOCK.fullmatch(part)
        if not m:
            raise ValueError(f"bad block {part!r} in {spec!r}")
        blocks.append((m[1], m[2], float(m[3]) if m[3] else 1.0))
    total = sum(w for _, _, w in blocks)
    return raw, [(n, l, w / total) for n, l, w in blocks]


def region_map(h, w, layout):
    """Region number of every pixel, shape (h, w)."""
    if layout is None:
        return np.zeros((h, w), np.intp)
    y, x = np.arange(h), np.arange(w)
    if layout == 'q2x2':
        return 2 * (y >= h // 2)[:, None] + (x >= w // 2)[None, :]
    return np.broadcast_to((3 * y // max(h, 1))[:, None], (h, w))


def count(codes, regions, nregions, nbins, weights=None):
    """Histogram of codes per region, shape (nregions, nbins). A region too
    small to have pixels (tiny images) gets the whole image's histogram."""
    hist = np.bincount((regions * nbins + codes).ravel(),
                       None if weights is None else weights.ravel(),
                       nregions * nbins).reshape(nregions, nbins)
    empty = hist.sum(axis=1) == 0
    if empty.any():
        hist[empty] = hist.sum(axis=0)
    return hist


def rgb_codes(img):
    q = img >> 6
    return (q[..., 0].astype(np.intp) << 4) | (q[..., 1] << 2) | q[..., 2]


def hsv_codes(img):
    hsv = np.asarray(Image.fromarray(img).convert('HSV')).astype(np.intp)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    color = 4 + (h * 12 >> 8) * 4 + (s >= 153) * 2 + (v >= 153)
    gray = 1 + np.minimum((v - 51) * 3 // 204, 2)
    return np.where(v < 51, 0, np.where(s < 51, gray, color))


def edge_hist(g, layout):
    h, w = g.shape
    n = REGIONS[layout]
    if h < 3 or w < 3:
        hist = np.zeros((n, 9))
        hist[:, 8] = 1
        return hist
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    edge = gx * gx + gy * gy > EDGE_THRESHOLD ** 2
    r = region_map(h - 2, w - 2, layout)
    # unsigned direction, 8 bins centred on 0, 22.5, 45... degrees so the
    # common horizontal and vertical edges each fall in a single bin; every
    # edge pixel splits its vote between the two nearest bins. Angles are
    # computed for edge pixels only, the slow part.
    er = r[edge] * 9
    pos = (np.arctan2(gy[edge], gx[edge]) % np.pi) * (8 / np.pi)
    lo = np.floor(pos)
    frac = pos - lo
    b0 = lo.astype(np.intp) % 8
    hist = (np.bincount(er + b0, 1 - frac, n * 9) +
            np.bincount(er + (b0 + 1) % 8, frac, n * 9)).reshape(n, 9)
    hist[:, 8] = np.bincount(r[~edge], minlength=n)
    empty = hist.sum(axis=1) == 0
    if empty.any():
        hist[empty] = hist.sum(axis=0)
    return hist


def lbp_hist(g, layout):
    h, w = g.shape
    n = REGIONS[layout]
    if h < 3 or w < 3:
        hist = np.zeros((n, 10))
        hist[:, 0] = 1
        return hist
    c = g[1:-1, 1:-1] + LBP_THRESHOLD
    bits = np.stack([g[dy:h - 2 + dy, dx:w - 2 + dx] >= c for dy, dx in LBP_RING])
    ones = bits.sum(axis=0)
    turns = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
    # "uniform" patterns (at most 2 changes around the ring) are edges,
    # corners, spots; the rest are lumped together
    codes = np.where(turns <= 2, ones, 9)
    return count(codes, region_map(h - 2, w - 2, layout), n, 10)


def size_values(osize):
    w, h = max(osize[0], 1), max(osize[1], 1)
    # 16x16 -> 0, 64 megapixels -> 1
    area = min(max((math.log2(w * h) - 8) / 18, 0), 1)
    aspect = min(max(math.log2(w / h), -2), 2) / 4
    return np.array([[area, aspect]])


def extract(img, osize, keys):
    """Raw counts of the blocks named in keys ((name, layout) pairs) for one
    thumbnail. Work shared between blocks (codes, gray image) is done once."""
    out, cache = {}, {}
    h, w = img.shape[:2]
    for name, layout in keys:
        if name == 'size':
            out[name, layout] = size_values(osize)
        elif name in ('rgb', 'hsv'):
            if name not in cache:
                cache[name] = rgb_codes(img) if name == 'rgb' else hsv_codes(img)
            out[name, layout] = count(cache[name], region_map(h, w, layout),
                                      REGIONS[layout], BINS[name])
        else:
            if 'gray' not in cache:
                cache['gray'] = img @ LUMA
            fn = edge_hist if name == 'edges' else lbp_hist
            out[name, layout] = fn(cache['gray'], layout)
    return out


def keys_of(specs):
    """Every (name, layout) block the given configs need."""
    keys = []
    for spec in specs:
        for name, layout, _ in parse(spec)[1]:
            if (name, layout) not in keys:
                keys.append((name, layout))
    return keys


def count_all(thumbs, osizes, keys):
    """Raw counts for many images: {(name, layout): (n, regions, bins)}."""
    n = len(thumbs)
    out = {k: np.empty((n, REGIONS[k[1]], BINS[k[0]]), np.float32) for k in keys}
    for i, (img, osize) in enumerate(zip(thumbs, osizes)):
        for k, v in extract(img, osize, keys).items():
            out[k][i] = v
    return out


def combine(counts, spec):
    """Feature matrix (n, dims) for a config, from count_all() output."""
    raw, blocks = parse(spec)
    parts = []
    for name, layout, weight in blocks:
        c = counts[name, layout].astype(np.float64)
        n, regions = c.shape[:2]
        if name == 'size':
            parts.append(c.reshape(n, -1) * math.sqrt(weight))
            continue
        p = c / c.sum(axis=2, keepdims=True)
        p = p * 100 if raw else np.sqrt(p)
        parts.append(p.reshape(n, -1) * math.sqrt(weight / regions))
    return np.concatenate(parts, axis=1)


def features(img, osize, spec):
    """Feature vector of one image, as the CLI would compute it."""
    keys = keys_of([spec])
    counts = {k: v[None] for k, v in extract(img, osize, keys).items()}
    return combine(counts, spec)[0].astype(np.float32)
