import numpy as np
from scipy.misc import imread

def loadImage(filename):
  with open(filename, 'rb') as img_file:
    image_array = imread(img_file)
    print image_array.shape


PATH_TO_IMAGE = 'data/'
FOLDER = 'train/'
IMAGE_NAME = 'sample'
IMAGE_POSTFIX = '.png'

LOAD_NUM_IMAGES = 3

for i in xrange(LOAD_NUM_IMAGES):
  loadImage(PATH_TO_IMAGE + IMAGE_NAME + str(i) + IMAGE_POSTFIX)


# Go through directory of images (train, valid, test)
  # For each image, 
    # extract array (using imread) --> save as 'data' in datadict
# Extract labels --> save as 'labels' in datadict
# Save datadict in pickle file

# data_utils can read in this pickle file for the class conv net implementation