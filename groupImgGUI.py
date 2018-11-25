import os
import sys
import shutil
import glob
import math
import argparse
import warnings
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

class K_means:

  def __init__(self, k=3, size=False, resample=32):
    self.k = k
    self.cluster = []
    self.data = []
    self.end = []
    self.i = 0
    self.size = size
    self.resample = resample

  def manhattan_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += abs( float(x1[i]) - float(x2[i]) )
    return s

  def euclidian_distance(self,x1,x2):
    s = 0.0
    for i in range(len(x1)):
      s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
    return s

  def read_image(self,im):
    if self.i >= self.k :
      self.i = 0
    try:
      img = Image.open(im)
      osize = img.size
      img.thumbnail((self.resample,self.resample))
      v = [float(p)/float(img.size[0]*img.size[1])*100  for p in np.histogram(np.asarray(img))[0]]
      if self.size :
        v += [osize[0], osize[1]]
      i = self.i
      self.i += 1
      return [i, v, im]
    except Exception as e:
      print("Error reading ",im,e)
      return [None, None, None]


  def generate_k_means(self):
    final_mean = []
    for c in range(self.k):
      partial_mean = []
      for i in range(len(self.data[0])):
        s = 0.0
        t = 0
        for j in range(len(self.data)):
          if self.cluster[j] == c :
            s += self.data[j][i]
            t += 1
        if t != 0 :
          partial_mean.append(float(s)/float(t))
        else:
          partial_mean.append(float('inf'))
      final_mean.append(partial_mean)
    return final_mean

  def generate_k_clusters(self,folder):
    pool = ThreadPool(cpu_count())
    result = pool.map(self.read_image, folder)
    pool.close()
    pool.join()
    self.cluster = [r[0] for r in result if r[0] != None]
    self.data = [r[1] for r in result if r[1] != None]
    self.end = [r[2] for r in result if r[2] != None]

  def rearrange_clusters(self):
    isover = False
    while(not isover):
      isover = True
      m = self.generate_k_means()
      for x in range(len(self.cluster)):
        dist = []
        for a in range(self.k):
          dist.append( self.manhattan_distance(self.data[x],m[a]) )
        _mindist = dist.index(min(dist))
        if self.cluster[x] != _mindist :
          self.cluster[x] = _mindist
          isover = False

class groupImgGUI(QWidget) :

	def __init__(self, parent = None) :
		
		super(groupImgGUI, self).__init__(parent)
		
		self.dir = None
		
		self.progressValue = 0
		
		self.createSettings()
		
		layout = QVBoxLayout()
		self.btn = QPushButton("Select folder")
		self.btn.clicked.connect(self.selectFolder)
		
		layout.addWidget(self.btn)
		
		self.check = QCheckBox("Settings")
		self.check.stateChanged.connect(self.state);
		
		self.runbtn = QPushButton("Run")
		self.runbtn.clicked.connect(self.run)
		
		self.progress = QProgressBar(self)
		self.progress.hide()
		
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
		layout.addRow(QLabel("k-means:"), self.kmeans)
		self.sample = QSpinBox()
		self.sample.setRange(32, 256)
		self.sample.setValue(128)
		self.sample.setSingleStep(2)
		layout.addRow(QLabel("Sample:"), self.sample)
		self.move = QCheckBox()
		layout.addRow(QLabel("Move:"), self.move)
		self.size = QCheckBox()
		layout.addRow(QLabel("Size:"), self.size)
		self.formGroupBox.hide()
		self.formGroupBox.setLayout(layout)
		
	def selectFolder(self) :
		QFileDialog.FileMode(QFileDialog.Directory)
		self.dir = QFileDialog.getExistingDirectory(self)
		self.btn.setText(self.dir or "Select folder")
		#QMessageBox.information(self, "fichier", 'oi')
		
	def state(self) :
		if self.check.isChecked() :
			self.formGroupBox.show()
		else:
			self.formGroupBox.hide()
	
	def run(self) :
		types = ('*.jpg', '*.JPG', '*.png', '*.jpeg')
		imagePaths = []

		folder = self.dir

		if not folder.endswith("/") :
			folder+="/"

		for files in types :
			imagePaths.extend(sorted(glob.glob(folder+files)))

		nimages = len(imagePaths)

		if nimages <= 0 :
			print("No images found!")
			exit()
		
		k = K_means(self.kmeans.value(),self.size.isChecked(),self.sample.value())

		k.generate_k_clusters(imagePaths)

		k.rearrange_clusters()

		for i in range(k.k) :
			try :
				os.makedirs(folder+str(i+1).zfill(self.kmeans.value()))
			except Exception as e :
				print("Folder already exists", e)

		action = shutil.copy
		if self.move.isChecked() :
			action = shutil.move

		for i in range(len(k.cluster)):
			action(k.end[i], folder+"/"+str(k.cluster[i]+1).zfill(self.kmeans.value())+"/")

	
def main():
   app = QApplication(sys.argv)
   ex = groupImgGUI()
   ex.show()
   sys.exit(app.exec_())
	
if __name__ == '__main__':
   main()
