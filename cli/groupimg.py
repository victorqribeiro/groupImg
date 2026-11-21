import os
import shutil
import glob
import math
import random
import argparse
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')

# distances this small are rounding errors, not real differences
EPSILON = 1e-9

class K_means:

  def __init__(self, k=3, size=False, resample=32):
    self.k = k
    self.cluster = []
    self.data = []
    self.end = []
    self.size = size
    self.resample = resample

  def read_image(self,im):
    try:
      img = Image.open(im)
      osize = img.size
      img.thumbnail((self.resample,self.resample))
      # joint RGB histogram over a fixed range, so colors are told apart and
      # images are comparable with each other
      pixels = np.asarray(img.convert('RGB')).reshape(-1, 3)
      hist = np.histogramdd(pixels, bins=4, range=[(0, 256)] * 3)[0]
      v = [float(p)/float(len(pixels))*100 for p in hist.flatten()]
      if self.size :
        v += [osize[0], osize[1]]
      pbar.update(1)
      return [v, im]
    except Exception as e:
      print("Error reading ",im,e)
      return [None, None]


  def generate_k_means(self):
    means = np.full((self.k, self.data.shape[1]), float('inf'))
    for c in range(self.k):
      members = self.data[self.cluster == c]
      if len(members) :
        means[c] = members.mean(axis=0)
    return means

  def generate_k_clusters(self,folder):
    pool = ThreadPool(cpu_count())
    result = pool.map(self.read_image, folder)
    pool.close()
    pool.join()
    self.data = np.array([r[0] for r in result if r[0] != None])
    self.end = [r[1] for r in result if r[1] != None]

  def distances(self,means):
    # squared euclidean distance from every image to every mean, one column
    # per mean. Written as |x|^2 - 2 x.m + |m|^2 so it is a single matrix
    # multiplication. The mean of a cluster is the point with the smallest
    # squared distance to its members, so assigning and updating agree.
    means = np.asarray(means, dtype=float)
    finite = np.isfinite(means).all(axis=1)
    m = np.where(finite[:, None], means, 0)
    dist = (self.data ** 2).sum(axis=1)[:, None] - 2 * self.data @ m.T + (m ** 2).sum(axis=1)
    # rounding can push a distance slightly below zero
    dist = np.maximum(dist, 0)
    # a mean of an empty cluster is infinitely far from everything
    dist[:, ~finite] = np.inf
    return dist

  def tolerance(self,means):
    # the matrix form above loses precision in proportion to the size of the
    # vectors, so "too close to call" has to scale with them
    means = means[np.isfinite(means).all(axis=1)]
    scale = (self.data ** 2).sum(axis=1) + (means ** 2).sum(axis=1).max()
    return EPSILON * (1 + scale)

  def init_clusters(self):
    # greedy k-means++ seeding: pick initial centers far apart from each
    # other, so distinct groups don't start out (and end up) sharing a
    # cluster. Each new center is the best of a few far-away candidates;
    # taking just one candidate often leaves two groups sharing a center
    trials = 2 + int(math.log(self.k))
    centers = [random.choice(self.data)]
    closest = ((self.data - centers[0]) ** 2).sum(axis=1)
    while len(centers) < self.k :
      if closest.sum() == 0 :
        break
      # the distance is already squared, as k-means++ wants
      picks = random.choices(range(len(self.data)), weights=closest, k=trials)
      options = [np.minimum(closest, ((self.data - self.data[p]) ** 2).sum(axis=1)) for p in picks]
      best = min(range(trials), key=lambda i: options[i].sum())
      centers.append(self.data[picks[best]])
      closest = options[best]
    self.cluster = self.distances(centers).argmin(axis=1)

  def fill_empty_clusters(self,m):
    # an empty cluster would never get images back, so give it the image
    # that is farthest from the center of its own cluster
    changed = False
    for c in range(self.k):
      counts = np.bincount(self.cluster, minlength=self.k)
      if counts[c] > 0 :
        continue
      dist = ((self.data - m[self.cluster]) ** 2).sum(axis=1)
      dist[counts[self.cluster] < 2] = 0
      far = dist.argmax()
      if dist[far] <= EPSILON :
        break
      self.cluster[far] = c
      changed = True
    return changed

  def rearrange_clusters(self,tries=10):
    # depending on where it starts, k-means can settle on a poor grouping,
    # so run it a few times and keep the tightest one
    if len(self.data) == 0 :
      return
    best = None
    for t in range(tries):
      self.init_clusters()
      self.run_k_means()
      m = self.generate_k_means()
      cost = ((self.data - m[self.cluster]) ** 2).sum()
      if best == None or cost < best[0] :
        best = (cost, self.cluster)
    self.cluster = best[1]

  def run_k_means(self):
    while True :
      m = self.generate_k_means()
      if self.fill_empty_clusters(m) :
        m = self.generate_k_means()
      dist = self.distances(m)
      cluster = dist.argmin(axis=1)
      # only move an image if it is clearly closer to another cluster, so
      # ties can't make it bounce between clusters forever
      current = dist[np.arange(len(dist)), self.cluster]
      stay = current - dist.min(axis=1) <= self.tolerance(m)
      cluster[stay] = self.cluster[stay]
      if (cluster == self.cluster).all() :
        break
      self.cluster = cluster

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to image folder")
ap.add_argument("-k", "--kmeans", type=int, default=5, help="how many groups")
ap.add_argument("-r", "--resample", type=int, default=128, help="size to resample the image by")
ap.add_argument("-s", "--size", default=False, action="store_true", help="use size to compare images")
ap.add_argument("-m", "--move", default=False, action="store_true", help="move instead of copy")
args = vars(ap.parse_args())
types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
imagePaths = []
folder = args["folder"]
for files in types :
	imagePaths.extend(sorted(glob.glob(os.path.join(folder+files))))
nimages = len(imagePaths)
nfolders = int(math.log(args["kmeans"], 10))+1
if nimages <= 0 :
	print("No images found!")
	exit()
if args["resample"] < 16 or args["resample"] > 256 :
	print("-r should be a value between 16 and 256")
	exit()
pbar = tqdm(total=nimages)
k = K_means(args["kmeans"],args["size"],args["resample"])
k.generate_k_clusters(imagePaths)
pbar.close()
k.rearrange_clusters()
for i in range(k.k) :
	currentFolder = os.path.join(folder, str(i+1).zfill(nfolders))
	try :
		os.makedirs(currentFolder)
	except FileExistsError:
		print("Folder '" + currentFolder + "' already exists")
	except Exception as e:
		print("An errror occurred creating folder '" + currentFolder + "': " + e)
action = shutil.copy2
if args["move"] :
	action = shutil.move
for i in range(len(k.cluster)):
	action(k.end[i], os.path.join(folder, str(k.cluster[i]+1).zfill(nfolders)+"/"))
