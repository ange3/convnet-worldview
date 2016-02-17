import numpy as np
from scipy.misc import imread

def loadImage(filename):
  with open(filename, 'rb') as img_file:
    image_array = imread(img_file)
    print image_array.shape

def load_npy_file(filename):
  '''
  Returns numpy array loaded from file
  '''
  arr = np.load(filename)
  return arr


def loadMediaEvalTruthLabels(filename):
  '''
  Reads in truth labels from specified filename
  '''
  with open(filename, 'r') as f:
    info_map = pickle.load(f)


# PATH_TO_IMAGE = 'data/'
# FOLDER = 'train/'
# IMAGE_NAME = 'sample'
# IMAGE_POSTFIX = '.png'

# LOAD_NUM_IMAGES = 3

# for i in xrange(LOAD_NUM_IMAGES):
#   loadImage(PATH_TO_IMAGE + IMAGE_NAME + str(i) + IMAGE_POSTFIX)


# Go through directory of images (train, valid, test)
  # For each image, 
    # extract array (using imread) --> save as 'data' in datadict
# Extract labels --> save as 'labels' in datadict
# Save datadict in pickle file

# data_utils can read in this pickle file for the class conv net implementation


# SMALL_DATA_000_PICKLE_FILE = 'data_maps/image_id_to_node_info_000_labels.pickle'
# y = loadMediaEvalData(SMALL_DATA_000_PICKLE_FILE)

# train_image_filenames = [f.split(".")[0] for f in listdir(path) if isfile(join(path, f))]