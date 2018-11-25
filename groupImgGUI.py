import sys
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

import groupimg

import os
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

class groupImgGUI(QWidget) :

	def __init__(self, parent = None) :
		
		super(groupImgGUI, self).__init__(parent)
		
		self.dir = None
		
		self.createSettings()
		
		layout = QVBoxLayout()
		self.btn = QPushButton("Select folder")
		self.btn.clicked.connect(self.selectFolder)
		
		layout.addWidget(self.btn)
		
		self.check = QCheckBox("Settings")
		self.check.stateChanged.connect(self.state);
		
		self.runbtn = QPushButton("Run")
		self.runbtn.clicked.connect(self.run)
		
		layout.addWidget(self.check)
		
		layout.addWidget(self.formGroupBox)
		
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
		'''
		print self.dir
		if self.check.isChecked()	:
			print self.kmeans.value()
			print self.sample.value()
			print self.move.isChecked()
			print self.size.isChecked()
		'''
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

		pbar = tqdm(total=nimages)

		k = groupimg.K_means(self.kmeans.value(),self.size.isChecked(),self.sample.value(),True)

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
