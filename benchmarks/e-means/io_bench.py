"""Time groupImg's per-image feature extraction on a typical camera photo."""
import time, numpy as np
from PIL import Image
import sys, os
d = sys.argv[1]
rng = np.random.default_rng(0)
arr = (rng.random((3000, 4000, 3)) * 255).astype(np.uint8)
p = os.path.join(d, "photo.jpg")
Image.fromarray(arr).save(p, quality=90)
t = time.time()
for _ in range(10):
    img = Image.open(p); img.thumbnail((128, 128))
    px = np.asarray(img.convert('RGB')).reshape(-1, 3)
    np.histogramdd(px, bins=4, range=[(0, 256)] * 3)
print(f"per 12MP image: {(time.time()-t)/10*1000:.0f} ms")
