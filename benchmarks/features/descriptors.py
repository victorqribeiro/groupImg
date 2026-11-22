"""Image descriptors for the groupImg feature benchmark.

A descriptor is a list of blocks. Each block is one histogram of the whole
image, or one per region of it:

  rgb     joint RGB histogram, 4 bins per channel (64). groupImg uses it now.
  hsv     52 bins: black, 3 grays, then 12 hues x 2 saturations x 2 values
  edges   gradient orientation: 8 unsigned directions plus 1 "flat" bin
  lbp     rotation-invariant uniform local binary patterns (10 bins)
  size    log area and log aspect ratio of the original image (2 values)
  freq    DCT frequency spectrum: the gray image is cut into 8x8 blocks, as
          in JPEG, and each block splits one vote between 14 frequency bands
          (u + v = 1..14 of its DCT coefficients) by its average coefficient
          size in each, or gives it to a 15th "flat" bin. Says how fine or coarse
          the detail is, wherever it is (15 bins)
  dct     layout: the 63 lowest-frequency DCT coefficients (top-left 8x8 but
          the average) of the gray image squashed to 32x32, scaled to length
          1. The arrangement of light and dark, ignoring brightness and
          contrast; close to a tiny blurred thumbnail (63 values)
  dctm    dct that ignores mirroring: flipping an image left to right
          only changes the sign of the coefficients with an odd horizontal
          frequency, so they are taken as absolute values (63 values)
  dctc    dct that ignores mirroring another way: images whose left half is
          darker than their right half (by the broadest left-right
          coefficient) are flipped, so all face the same way (63 values)
  phash   the same coefficients reduced to above / below their median, as
          the perceptual hash pHash does, as +-1/sqrt(63) (63 values)

"@q2x2" computes a block per quadrant, "@bands3" per horizontal third.
dct, dctm, dctc and phash describe the whole image and take no region.

A config joins blocks with "+", each with an optional weight:
"rgb*0.4+rgb@bands3*0.35+edges@bands3*0.25". Weights default to 1 and are
rescaled to sum to 1.

combine() divides every histogram by its pixel count and takes the square
root. Each histogram then has length 1, and the squared Euclidean distance
between two of them is twice their squared Hellinger distance (0 to 2). A
block's weight is its share of the distance, split evenly between its
regions. The size block is already on a 0-1 scale and is used as is.
The dct blocks and phash are already vectors of length 1 (squared distance
0 to 4) and are only weighted. Neither the size block nor these can be
square-rooted, as dct has negative values.

The prefix "raw:" skips the square root and keeps histograms in percent,
exactly as groupImg does now.
"""
import math
import re
import numpy as np
from PIL import Image, ImageOps, ExifTags

BINS = {'rgb': 64, 'hsv': 52, 'edges': 9, 'lbp': 10, 'size': 2,
        'freq': 15, 'dct': 63, 'dctm': 63, 'dctc': 63, 'phash': 63}
# blocks made from the layout coefficients, which describe the whole image
LAYOUT_BLOCKS = ('dct', 'dctm', 'dctc', 'phash')
# blocks that are vectors already, not histograms to normalize
VECTORS = ('size',) + LAYOUT_BLOCKS
REGIONS = {None: 1, 'q2x2': 4, 'bands3': 3}
# a gradient below this (in 0-255 levels across 2 pixels) counts as flat
EDGE_THRESHOLD = 20.0
# a neighbor must be this much brighter than the center to set its LBP bit,
# so JPEG noise in flat areas doesn't make random patterns
LBP_THRESHOLD = 3.0
# neighbors of a 3x3 window in circular order, as offsets from its corner
LBP_RING = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
# an 8x8 block whose pixels vary less than this (standard deviation, in
# 0-255 levels) counts as flat, so JPEG noise isn't taken for fine detail
FLAT_THRESHOLD = 3.0
# the gray image is squashed to this size for the dct and phash layouts
DCT_SIZE = 32
# and they keep this many lowest frequencies in each direction
DCT_KEEP = 8
# a layout whose broad shapes vary less than this (standard deviation, in
# 0-255 levels) is a constant image plus rounding (about 0.00002), not an
# image: the least varied of the 2,000 test images has 0.375
LAYOUT_FLAT = 0.01


def dct_matrix(n):
    """Orthonormal DCT-II: dct_matrix(n) @ x transforms the columns of x,
    and C @ img @ C.T a whole image."""
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    c = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
    c[0] /= math.sqrt(2)
    return c.astype(np.float32)


DCT8 = dct_matrix(8)
DCT_BIG = dct_matrix(DCT_SIZE)
# frequency band (u + v - 1, 0-13) of each of the 64 coefficients of an 8x8
# block, as a 64 x 14 matrix that averages a block's coefficients per band.
# Averages, not sums, as the middle bands have up to 8 coefficients and the
# ends 1: noise then spreads evenly. The block average (u = v = 0) belongs to
# no band
_uv = np.add.outer(np.arange(8), np.arange(8)).ravel()
FREQ_BANDS = np.zeros((64, 14), np.float32)
FREQ_BANDS[np.arange(1, 64), _uv[1:] - 1] = 1
FREQ_BANDS /= FREQ_BANDS.sum(axis=0)
# which of the layout coefficients (row-major top-left DCT_KEEP x DCT_KEEP,
# without the average) have an odd horizontal frequency: mirroring the image
# left to right multiplies exactly these by -1. The first is the broadest
# left-right one, positive when the left half is brighter
ODD_U = (np.arange(1, DCT_KEEP * DCT_KEEP) % DCT_KEEP) % 2 == 1

BLOCK = re.compile(r'(rgb|hsv|edges|lbp|size|freq|dctm|dctc|dct|phash)'
                   r'(?:@(q2x2|bands3))?(?:\*([0-9.]+))?')


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
        if m[1] in LAYOUT_BLOCKS and m[2]:
            raise ValueError(f"{m[1]} describes the whole image, not @{m[2]}, in {spec!r}")
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


def freq_votes(g):
    """Votes of the 8x8 blocks of a gray image for the freq histogram:
    shape (rows, cols, 15), each block's votes summing to 1. Pixels past
    the last whole block are left out."""
    rows, cols = g.shape[0] // 8, g.shape[1] // 8
    blocks = g[:rows * 8, :cols * 8].reshape(rows, 8, cols, 8).swapaxes(1, 2)
    coef = np.einsum('ui,rcij,vj->rcuv', DCT8, blocks, DCT8).reshape(rows, cols, 64)
    # the DCT is orthonormal, so the coefficients but the average hold the
    # block's variance times 64
    flat = (coef[..., 1:] ** 2).sum(axis=2) < 64 * FLAT_THRESHOLD ** 2
    # magnitudes, not squares, so the strongest band doesn't take everything
    detail = np.abs(coef) @ FREQ_BANDS
    total = detail.sum(axis=2, keepdims=True)
    votes = np.zeros((rows, cols, 15), np.float32)
    votes[..., :14] = np.where(flat[..., None], 0, detail / np.maximum(total, 1e-12))
    votes[..., 14] = flat
    return votes


def freq_hist(votes, layout):
    n = REGIONS[layout]
    rows, cols = votes.shape[:2]
    if rows == 0 or cols == 0:
        # smaller than one block: no detail to measure
        hist = np.zeros((n, 15))
        hist[:, 14] = 1
        return hist
    r = region_map(rows, cols, layout).ravel()
    hist = np.zeros((n, 15))
    np.add.at(hist, r, votes.reshape(-1, 15))
    empty = hist.sum(axis=1) == 0
    if empty.any():
        hist[empty] = hist.sum(axis=0)
    return hist


def layout_coefficients(g):
    """The 63 lowest-frequency DCT coefficients of a gray image squashed to
    DCT_SIZE x DCT_SIZE, without the average."""
    small = np.asarray(Image.fromarray(np.ascontiguousarray(g, np.float32))
                       .resize((DCT_SIZE, DCT_SIZE), Image.BOX))
    coef = DCT_BIG @ small @ DCT_BIG.T
    return coef[:DCT_KEEP, :DCT_KEEP].ravel()[1:]


def layout_is_flat(coef):
    # the coefficients hold the variance of the broad shapes times
    # DCT_SIZE^2. Only a constant image counts as flat. A weak layout is
    # scaled up like any other: fine regular textures (dots, studs) average
    # out to almost nothing at 32x32, and their blown-up noise points in a
    # random direction, as far from everything as unrelated images are.
    # Treating them as flat made them everyone's nearest image (see README)
    return (coef.astype(np.float64) ** 2).sum() < (DCT_SIZE * LAYOUT_FLAT) ** 2


def dct_values(coef):
    # length 1, so contrast doesn't count, only the arrangement. A constant
    # image has no arrangement and stays 0: squared distance 1 from every
    # image, closer than unrelated images (2 on average), so blank images
    # pull a little toward everything. They are rare enough to accept
    if layout_is_flat(coef):
        return np.zeros((1, len(coef)))
    return (coef / np.sqrt((coef.astype(np.float64) ** 2).sum()))[None]


def dctm_values(coef):
    # a mirrored copy has the same absolute values. What is lost is how the
    # odd coefficients' signs relate to each other, so some arrangements
    # that are not mirror images of each other come out the same too
    return dct_values(np.where(ODD_U, np.abs(coef), coef))


def dctc_values(coef):
    # keeps every sign relation, but an image that is nearly as bright left
    # as right can fall either way, away from a copy that fell the other way
    flip = ODD_U & (coef[ODD_U][0] < 0)
    return dct_values(np.where(flip, -coef, coef))


LAYOUT_VALUES = {'dct': dct_values, 'dctm': dctm_values, 'dctc': dctc_values}


def phash_values(coef):
    if layout_is_flat(coef):
        return np.zeros((1, len(coef)))
    bits = coef > np.median(coef)
    return (np.where(bits, 1.0, -1.0) / math.sqrt(len(coef)))[None]


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
            g = cache['gray']
            if name == 'freq':
                if 'freq' not in cache:
                    cache['freq'] = freq_votes(g)
                out[name, layout] = freq_hist(cache['freq'], layout)
            elif name in LAYOUT_BLOCKS:
                if 'layout' not in cache:
                    cache['layout'] = layout_coefficients(g)
                fn = LAYOUT_VALUES.get(name, phash_values)
                out[name, layout] = fn(cache['layout'])
            else:
                fn = edge_hist if name == 'edges' else lbp_hist
                out[name, layout] = fn(g, layout)
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
        if name in VECTORS:
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
