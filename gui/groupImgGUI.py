import os
import sys
import shutil
import glob
import math
import random
import warnings
import numpy as np
from PIL import Image, ImageOps, ExifTags
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore')

# distances this small are rounding errors, not real differences
EPSILON = 1e-9
# k-means always settles; this only stops rounding from keeping it busy
MAX_ITERATIONS = 1000

# How much each part of an image's description counts in the distance
# between two images: color over the whole image, color in each horizontal
# third (sky above, ground below), edges and texture in each third. They add
# up to 1. benchmarks/features/ has how they were chosen.
WEIGHTS = (0.25, 0.25, 0.25, 0.25)
# with -s, how much the image's size counts next to them
SIZE_WEIGHT = 1.0
# a change in brightness smaller than this across 2 pixels is not an edge
EDGE_THRESHOLD = 20
# a neighbor must be this much brighter to count in a texture pattern, so
# JPEG noise in flat areas doesn't make random patterns
TEXTURE_THRESHOLD = 3
# the 8 neighbors of a pixel in circular order, as offsets in a 3x3 window
RING = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

def per_band(codes, nbins, weights=None):
  # one histogram of the codes in each horizontal third of the image
  band = (3 * np.arange(len(codes)) // len(codes))[:, None]
  index = (band * nbins + codes).ravel()
  hist = np.bincount(index, None if weights is None else weights.ravel(), 3 * nbins)
  return hist.reshape(3, nbins)

def edges(gray):
  # direction of the change in brightness at each pixel, in 8 bins centred
  # on 0, 22.5, 45... degrees so horizontal and vertical edges each fall in
  # one bin, and a 9th bin for flat pixels. An edge pixel splits its vote
  # between the two nearest directions. Angles are only computed for edge
  # pixels, as they are the slow part
  gx = gray[1:-1, 2:] - gray[1:-1, :-2]
  gy = gray[2:, 1:-1] - gray[:-2, 1:-1]
  edge = gx * gx + gy * gy > EDGE_THRESHOLD ** 2
  band = np.broadcast_to((3 * np.arange(len(edge)) // len(edge))[:, None], edge.shape)
  pos = (np.arctan2(gy[edge], gx[edge]) % np.pi) * (8 / np.pi)
  low = np.floor(pos)
  frac = pos - low
  b0 = low.astype(np.intp) % 8
  b = band[edge] * 9
  hist = np.bincount(b + b0, 1 - frac, 27) + np.bincount(b + (b0 + 1) % 8, frac, 27)
  hist = hist.reshape(3, 9)
  hist[:, 8] = np.bincount(band[~edge], minlength=3)
  return hist

def texture(gray):
  # local binary patterns: which of its 8 neighbors are brighter than each
  # pixel. Patterns with at most 2 changes around the ring (edges, corners,
  # spots) count by how many neighbors are brighter (0-8), so a rotated
  # pattern counts the same; all others count together (9)
  h, w = gray.shape
  center = gray[1:-1, 1:-1] + TEXTURE_THRESHOLD
  bits = np.stack([gray[y:h - 2 + y, x:w - 2 + x] >= center for y, x in RING])
  turns = (bits != np.roll(bits, 1, axis=0)).sum(axis=0)
  return per_band(np.where(turns <= 2, bits.sum(axis=0), 9), 10)

def describe(pixels, size=None):
  # feature vector of an image, from a height x width x 3 array of RGB
  # values: histograms of color (4 levels per channel, 64 bins), edges and
  # texture, each divided by its pixel count and square-rooted. The squared
  # distance between two images is then twice their Hellinger distance,
  # which unlike plain histograms doesn't let one big background color
  # outweigh everything else. With -s, size is the log of the area and of
  # the aspect ratio of the original image
  q = pixels >> 6
  color = per_band((q[..., 0].astype(np.intp) << 4) | (q[..., 1] << 2) | q[..., 2], 64)
  parts = [color.sum(axis=0, keepdims=True), color]
  gray = pixels @ LUMA
  if min(gray.shape) < 3 :
    # too small to have edges or texture: all flat
    parts += [np.tile(np.eye(9)[8], (3, 1)), np.tile(np.eye(10)[0], (3, 1))]
  else :
    parts += [edges(gray), texture(gray)]
  v = []
  for hist, weight in zip(parts, WEIGHTS):
    # a third too small to have pixels (tiny images) gets the whole image's
    empty = hist.sum(axis=1) == 0
    hist[empty] = hist.sum(axis=0)
    v.append(np.sqrt(hist / hist.sum(axis=1, keepdims=True)) * math.sqrt(weight / len(hist)))
  if size :
    w, h = max(size[0], 1), max(size[1], 1)
    area = min(max((math.log2(w * h) - 8) / 18, 0), 1)  # 16x16 -> 0, 64 MP -> 1
    aspect = min(max(math.log2(w / h), -2), 2) / 4
    v.append(np.array([area, aspect]) * math.sqrt(SIZE_WEIGHT))
  return np.concatenate([p.ravel() for p in v]).astype(np.float32)

class K_means:

  def __init__(self, k=3, size=False, resample=128):
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
      # thumbnail() decodes JPEGs at a reduced size, so this is cheap. The
      # EXIF data survives it, so rotating afterwards only moves a small image
      img.thumbnail((self.resample,self.resample))
      if img.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8) :
        osize = osize[::-1]
      img = ImageOps.exif_transpose(img)
      if img.mode in ('RGBA', 'LA', 'PA') or 'transparency' in img.info :
        # transparent areas would count as whatever color hides under them,
        # often black; put them on white, as most viewers show them
        img = img.convert('RGBA')
        img = Image.alpha_composite(Image.new('RGBA', img.size, 'white'), img)
      v = describe(np.asarray(img.convert('RGB')), osize if self.size else None)
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
    self.data = np.array([r[0] for r in result if r[0] is not None])
    self.end = [r[1] for r in result if r[1] is not None]

  def prepare(self):
    # the squared length of every image, which every distance needs and
    # which doesn't change, and the rounding error to expect: images from
    # read_image are float32, which rounds more than float64
    self.sqlen = (self.data.astype(np.float64) ** 2).sum(axis=1)
    self.epsilon = max(EPSILON, 64 * np.finfo(self.data.dtype).eps)

  def distances(self,means):
    # squared euclidean distance from every image to every mean, one column
    # per mean. Written as |x|^2 - 2 x.m + |m|^2 so it is a single matrix
    # multiplication, done in the images' own precision: converting them
    # all to float64 on every call costs more than the multiplication. The
    # mean of a cluster is the point with the smallest squared distance to
    # its members, so assigning and updating agree.
    means = np.asarray(means, dtype=float)
    finite = np.isfinite(means).all(axis=1)
    m = np.where(finite[:, None], means, 0)
    dist = self.sqlen[:, None] - 2 * (self.data @ m.T.astype(self.data.dtype)) + (m ** 2).sum(axis=1)
    # rounding can push a distance slightly below zero
    dist = np.maximum(dist, 0)
    # a mean of an empty cluster is infinitely far from everything
    dist[:, ~finite] = np.inf
    return dist

  def tolerance(self,means):
    # the matrix form above loses precision in proportion to the size of the
    # vectors, so "too close to call" has to scale with them
    means = means[np.isfinite(means).all(axis=1)]
    scale = self.sqlen + (means ** 2).sum(axis=1).max()
    return self.epsilon * (1 + scale)

  def distances_to(self,rows):
    # squared distance from every image to the images at rows, one column
    # each. Differences within rounding error count as 0, so a duplicate of
    # a center is never taken for a new one
    dist = self.sqlen[:, None] - 2 * (self.data @ self.data[rows].T) + self.sqlen[rows]
    close = dist <= self.epsilon * (1 + self.sqlen[:, None] + self.sqlen[rows])
    return np.where(close, 0, dist)

  def init_clusters(self):
    # greedy k-means++ seeding: pick initial centers far apart from each
    # other, so distinct groups don't start out (and end up) sharing a
    # cluster. Each new center is the best of a few far-away candidates;
    # taking just one candidate often leaves two groups sharing a center
    trials = 2 + int(math.log(self.k))
    centers = [random.randrange(len(self.data))]
    closest = self.distances_to(centers)[:, 0]
    while len(centers) < self.k :
      if closest.sum() == 0 :
        break
      # the distance is already squared, as k-means++ wants
      picks = random.choices(range(len(self.data)), weights=closest, k=trials)
      options = np.minimum(closest[:, None], self.distances_to(picks))
      best = options.sum(axis=0).argmin()
      centers.append(picks[best])
      closest = options[:, best]
    self.cluster = self.distances(self.data[centers]).argmin(axis=1)

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
    self.prepare()
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
    for i in range(MAX_ITERATIONS) :
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

class groupImgGUI(QWidget) :

	def __init__(self, parent = None) :		
		super(groupImgGUI, self).__init__(parent)	
		self.dir = None		
		self.progressValue = 0		
		self.createSettings()		
		layout = QVBoxLayout()
		self.btn = QPushButton("Select folder")
		self.btn.clicked.connect(self.selectFolder)			
		self.check = QCheckBox("Settings")
		self.check.stateChanged.connect(self.state);		
		self.runbtn = QPushButton("Run")
		self.runbtn.clicked.connect(self.run)	
		self.progress = QProgressBar(self)
		self.progress.hide()
		layout.addWidget(self.btn)
		layout.addWidget(self.check)		
		layout.addWidget(self.formGroupBox)	
		layout.addWidget(self.progress)	
		layout.addWidget(self.runbtn)	
		self.setMinimumSize(300,300)
		self.setLayout(layout)
		self.setWindowTitle("groupImg - GUI")

	def createSettings(self) :
		self.formGroupBox = QGroupBox("Settings")
		layout = QFormLayout()
		self.kmeans = QSpinBox()
		self.kmeans.setRange(3,15)
		self.kmeans.setValue(3)
		self.sample = QSpinBox()
		self.sample.setRange(32, 256)
		self.sample.setValue(128)
		self.sample.setSingleStep(2)
		self.move = QCheckBox()
		self.size = QCheckBox()
		layout.addRow(QLabel("N. Groups:"), self.kmeans)
		layout.addRow(QLabel("Resample:"), self.sample)
		layout.addRow(QLabel("Move:"), self.move)
		layout.addRow(QLabel("Size:"), self.size)
		self.formGroupBox.hide()
		self.formGroupBox.setLayout(layout)
		
	def selectFolder(self) :
		QFileDialog.FileMode(QFileDialog.Directory)
		self.dir = QFileDialog.getExistingDirectory(self)
		self.btn.setText(self.dir or "Select folder")
		
	def state(self) :
		if self.check.isChecked() :
			self.formGroupBox.show()
		else:
			self.formGroupBox.hide()
	
	def disableButton(self) :
		self.runbtn.setText("Working...")
		self.runbtn.setEnabled(False)
	
	def enableButton(self) :		
		self.runbtn.setText("Run")
		self.runbtn.setEnabled(True)
			
	def run(self) :
		self.disableButton()
		types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
		imagePaths = []
		folder = self.dir
		if not folder.endswith("/") :
			folder+="/"
		for files in types :
			imagePaths.extend(sorted(glob.glob(folder+files)))
		nimages = len(imagePaths)
		nfolders = int(math.log(self.kmeans.value(), 10))+1		
		if nimages <= 0 :
			QMessageBox.warning(self, "Error", 'No images found!')
			self.enableButton()
			return		
		k = K_means(self.kmeans.value(),self.size.isChecked(),self.sample.value())
		k.generate_k_clusters(imagePaths)
		k.rearrange_clusters()
		for i in range(k.k) :
			try :
				os.makedirs(folder+str(i+1).zfill(nfolders))
			except Exception as e :
				print("Folder already exists", e)
		action = shutil.copy
		if self.move.isChecked() :
			action = shutil.move
		for i in range(len(k.cluster)):
			action(k.end[i], folder+"/"+str(k.cluster[i]+1).zfill(nfolders)+"/")
		QMessageBox.information(self, "Done", 'Done!')
		self.enableButton()
	
def main():
   app = QApplication(sys.argv)
   groupimg = groupImgGUI()
   groupimg.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
