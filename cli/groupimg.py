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

  def euclidian_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
    return s

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
    # manhattan distance from every image to every mean, one column per mean
    return np.stack([np.abs(self.data - m).sum(axis=1) for m in means], axis=1)

  def init_clusters(self):
    # k-means++ seeding: pick initial centers far apart from each other, so
    # distinct groups don't start out (and end up) sharing a cluster
    centers = [random.choice(self.data)]
    closest = self.distances(centers)[:, 0]
    while len(centers) < self.k :
      if closest.sum() == 0 :
        break
      centers.append(random.choices(self.data, weights=closest ** 2)[0])
      closest = np.minimum(closest, np.abs(self.data - centers[-1]).sum(axis=1))
    self.cluster = self.distances(centers).argmin(axis=1)

  def fill_empty_clusters(self,m):
    # an empty cluster would never get images back, so give it the image
    # that is farthest from the center of its own cluster
    changed = False
    for c in range(self.k):
      counts = np.bincount(self.cluster, minlength=self.k)
      if counts[c] > 0 :
        continue
      dist = np.abs(self.data - m[self.cluster]).sum(axis=1)
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
      cost = np.abs(self.data - m[self.cluster]).sum()
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
      stay = current - dist.min(axis=1) <= EPSILON
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
