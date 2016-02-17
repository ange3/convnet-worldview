import os
from PIL import Image
import random
from sets import Set
import pickle

directories = ['train', 'test', 'validation']
for d in directories:
	files = os.listdir(d)
	for f in files:
		img = Image.open(d + '/' + f)
		img = img.resize((250,150))
		img.save('data_250by150/' + f)
