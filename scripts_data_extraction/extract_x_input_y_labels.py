'''
This file pre-processes the data images and labels.
'''


from os import listdir
from os.path import isfile, join
import pickle, csv

import numpy as np
from scipy.misc import imread

# Naming files
X_INPUT_NPY_FILENAME = 'x_input'
Y_OUTPUT_NPY_FILENAME = 'y_labels'
MAP_LABELS_FILENAME = 'country_name_class_index_map.pickle'


# FOR 000 SMALL DATASET
# PATH_TO_PLACE_TASK_DATA_ALL = '../data/000_small/'
# SAVE_TO_DATA_MAPS_FOLDER = '../data_maps/000_small/'

# FOR 000 SMALL DATASET, 50x30
# PATH_TO_PLACE_TASK_DATA_ALL = '../data/000_small_50by30/'
# SAVE_TO_DATA_MAPS_FOLDER = '../data_maps/000_small_50by30/'

# FOR 000 SMALL DATASET, 48x32
# PATH_TO_PLACE_TASK_DATA_ALL = '../data/000_small_48by32/'
# SAVE_TO_DATA_MAPS_FOLDER = '../data_maps/000_small_48by32/'
# ALL_TRUTH_LABELS_FILENAME = 'image_id_to_node_info_000_small.pickle'

# FOR SUBSET_DATASET_5, 48x32
PATH_TO_PLACE_TASK_DATA_ALL = '/Users/angelasy/Dropbox/cs231n/subset_data_48by32_5/'
SAVE_TO_DATA_MAPS_FOLDER = '../data_maps/subset_48by32_5/'
ALL_TRUTH_LABELS_FILENAME = 'image_id_to_node_info_subset_data_5.pickle'
X_BATCHSIZE = 30000

IMG_HEIGHT = 48
IMG_WIDTH = 32
NUM_CHANNELS = 3

# Full Data Set
PATH_TO_PLACE_TASK_DATA = '../../mediaeval_placing_task_2015_data/'

TRUTH_LABELS_FILENAME = 'mediaeval2015_placing_locale_train'


def get_image_filenames(path):
  '''
  Returns a list of filenames in the folder of the specified path
  '''

  # NOTE: Make sure to delete any hidden files in this folder (such as .DS_STORE) that may get added to the image_filenames list
  image_filenames = [f.split(".")[0] for f in listdir(path) if isfile(join(path, f))]  # remove the ".png" part of filename
  # print 'image_filenames = {}'.format(image_filenames)
  return image_filenames

def open_labels_data(image_filename_list = [], pickle_filename = None):
  '''
  1) Opens file with truth labels and saves info on images in a map. Loads only image data pertaining to given image filenames if a list of images is specified.
    { image ID: (node_id, node_name, long, lat) }

  Note: ignore data on videos

  2) Saves data to a pickle file
  '''
  image_info_map = {}
  file_path = PATH_TO_PLACE_TASK_DATA + TRUTH_LABELS_FILENAME

  # 1) Open ground truth data
  print 'INFO: Loading ground truth data...'
  print 'Loading labels for {} images'.format(len(image_filename_list))
  ncount = 0
  added_to_map_count = 0
  with open(file_path) as tsv:
    for line in reversed(list(csv.reader(tsv, dialect="excel-tab"))):  # using reversed list since labels for folder 5 subset data images are at the end of the labels file
      if ncount % 20000 == 0:
        print 'processed {} * 20k photos'.format(ncount/20000)
        print 'map count: {}'.format(added_to_map_count)
      ncount += 1
      obj_id = line[0]
      if obj_id not in image_filename_list:  # check if we are using a shortened list of images
        # print 'Skipped: Not in given list'
        continue
      obj_type = line[1]
      if obj_type == 1:  # ignore video data
        continue
      image_info_map[obj_id] = line[2:]  # [year, node_id, node_name, long, lat]
      # print '{} added to map'.format(obj_id)
      added_to_map_count += 1

      if added_to_map_count == len(image_filename_list):
        print 'Found all the {} needed labels!'.format(added_to_map_count)
        break

  print 'Created map with {} objects.'.format(len(image_info_map))

  # 2) Save to pickle file
  print 'INFO: Saving to pickle file...'
  if pickle_filename: 
    with open(pickle_filename, 'w') as f:
      pickle.dump(image_info_map, f)

    # test pickle file 
    # obj_id = '00cb928aff9719f0d57c21e371288b'  # 000_small_48by32
    obj_id = 'dc2a1cf13b396bc3c068f3b170d5a342'  # subset_data_48by32_5
    print 'obj_id = {}'.format(obj_id)
    print 'original value = {}'.format(image_info_map[obj_id])
    with open(pickle_filename, 'r') as f:
      test_map = pickle.load(f)
      print 'pickled value = {}'.format(test_map[obj_id])

  return image_info_map


def load_images_data(image_filenames_list, base_save_filename, batchsize = -1):
  '''
  Loads all images from list of filenames and saves them to a numpy file.
  Returns:
  1) If not processing by batch, a list of image filenames that passed the spec test (correct width, height, channel size)
  2) If processing by batch, a map {batch_count: [list of filenames added to that X batch] }
  '''
  print 'Loading folder with {} images.'.format(len(image_filenames_list))

  # Create X array
  num_images_total = len(image_filenames_list)
  if batchsize != -1 and batchsize < num_images_total:
    all_image_data = np.empty((batchsize, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS))
  else:
    all_image_data = np.empty((num_images_total, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS))

  # Set up data struct to store list of image filenames added to X input data
  if batchsize != -1: 
    filenames_list_added_to_X_per_batch = {}
  else:
    filenames_list_added_to_X = []

  # Set up variables
  img_count = 0
  batch_count = 0
  total_img_count = 0

  # Loop through all images and add them to X
  for img_index, image_filename in enumerate(image_filenames_list):
    if batchsize != -1 and img_count == batchsize:
      # Save data
      print '-- SAVE: X as np file - Batch', batch_count
      np_save_filename = base_save_filename + '_' + str(batch_count) + '.npy'
      print np_save_filename
      np.save(np_save_filename, all_image_data)

      # Testing
      arr = np.load(np_save_filename)
      print '-- Testing: Loading np file'
      print 'Loaded array with shape {}'.format(arr.shape)

      # Update for next loop
      img_count = 0
      batch_count += 1

      if batch_count < num_images_total / batchsize:
        all_image_data = np.empty((batchsize, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS))
      else:
        num_images_remaining = num_images_total % batchsize
        all_image_data = np.empty((num_images_remaining, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS))

    if img_index % 100 == 0:
      print 'Processed {} *100 images.'.format(img_index/100)
    if image_filename == "":  # ignore error filenames
      continue
    filename = PATH_TO_PLACE_TASK_DATA_ALL + image_filename + '.png'
    with open(filename, 'rb') as img_file:
      # Read image
      image_array = imread(img_file)  # (W, H, C)

      # Skip image if it does not have 3 channels
      if image_array.shape != (IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS):   
        continue

      # Insert image data in X array
      all_image_data[img_count] = image_array
      img_count += 1
      total_img_count += 1

      # Add filename to list of images added to X (keeping track so we can load Y labels later)
      if batchsize != -1:
        if batch_count not in filenames_list_added_to_X_per_batch:
          filenames_list_added_to_X_per_batch[batch_count] = []
        filenames_list = filenames_list_added_to_X_per_batch[batch_count]
        filenames_list.append(image_filename)
      else:
        filenames_list_added_to_X.append(image_filename)

  print 'Num images loaded that passed spec: {}'.format(total_img_count)
  all_image_data = all_image_data[:img_count]
  print 'X.shape: {}'.format(all_image_data.shape)

  # Save data array
  print '-- SAVE: Saving X as np file'
  if batchsize != -1:
    np_save_filename = base_save_filename + '_' + str(batch_count) + '.npy'
  else:
    np_save_filename = base_save_filename + '.npy'
  print np_save_filename
  np.save(np_save_filename, all_image_data)

  # Testing saved data array
  arr = np.load(np_save_filename)
  print 'INFO: Loading np file'
  print 'Loaded array with shape {}'.format(arr.shape)

  # Return list of filenames or map of lists (if using batches)
  if batchsize != -1:
    return filenames_list_added_to_X_per_batch
  else:
    return filenames_list_added_to_X

def extract_y_from_filenames_and_save(image_filenames_list, image_info_map, country_name_class_index_map, np_save_filename):
  '''
  Creates matrix with extracted class labels of the given list of filenames and saves file.
  Note: Mapping of country name to class label stored in given map
  '''
  Y = np.empty((len(image_filenames_list)), dtype=np.int32)  # using type int32 because theano ivector for truth labels expects a vector of int32 values later on
  for index, img_name in enumerate(image_filenames_list):
    country_name = image_info_map[img_name][2].split('@')[0]  # take country name from node name in index 4
    if country_name not in country_name_class_index_map:
      country_name_class_index_map[country_name] = len(country_name_class_index_map)
    class_index = country_name_class_index_map[country_name]
    Y[index] = class_index

  # Save data
  print '-- SAVE: Saving Y vector as np file'
  print np_save_filename
  print 'Y.shape', Y.shape
  np.save(np_save_filename, Y)
  print 'Y sample: ', Y[:10]

  # Testing data
  arr = np.load(np_save_filename)
  print 'INFO: Loading np file'
  print 'Loaded array with shape {}'.format(arr.shape)


def get_truth_labels(image_filenames_list, pickled_all_info_file, y_base_save_filename, pickle_class_map_filename, using_batches = False):
  '''
  Extract Y in correct order (based on image_filenames_list).
  Save Y as numpy file
  '''
  with open(pickled_all_info_file, 'r') as f:
    image_info_map = pickle.load(f)
    country_name_class_index_map = {}

    # Extract y labels from list of filenames
    if using_batches:  # map
      for batch_count, file_list in image_filenames_list.items():
        np_save_filename = y_base_save_filename + '_' + str(batch_count) + '.npy'
        Y = extract_y_from_filenames_and_save(file_list, image_info_map, country_name_class_index_map, np_save_filename)
    else:  # list
      np_save_filename = y_base_save_filename + '.npy'
      Y = extract_y_from_filenames_and_save(image_filenames_list, image_info_map, country_name_class_index_map, np_save_filename)

    print '-- SAVE: Saving country name to class index map to pickle file'
    print pickle_class_map_filename
    print 'Number of country class labels:', len(country_name_class_index_map)
    with open(pickle_class_map_filename, 'w') as f:
      pickle.dump(country_name_class_index_map, f)




if __name__ == "__main__":
  image_filenames = get_image_filenames(PATH_TO_PLACE_TASK_DATA_ALL)
  print 'Data Set', PATH_TO_PLACE_TASK_DATA_ALL

  # For testing, limit number of images
  # Note: Can control X and Y arrays by manipulating image_filenames list
  # CLIP_NUM_IMAGES = 10
  # image_filenames = image_filenames[:CLIP_NUM_IMAGES]

  # 0) Load labels data from mediaeval train file and saves to pickle file --> Just run this once - saves labels in a pickle file which can later be used for processing truth labels for all images in same dataset
  # open_labels_data(image_filenames, pickle_filename = SAVE_TO_DATA_MAPS_FOLDER + ALL_TRUTH_LABELS_FILENAME)

  # 1) Get input data X
  final_image_filenames = load_images_data(image_filenames, SAVE_TO_DATA_MAPS_FOLDER + X_INPUT_NPY_FILENAME, batchsize = X_BATCHSIZE) # Pass in npy file name to save to

  # 2) Get truth labels Y
  if X_BATCHSIZE != -1:
    using_batches = True

  get_truth_labels(final_image_filenames, SAVE_TO_DATA_MAPS_FOLDER + ALL_TRUTH_LABELS_FILENAME, SAVE_TO_DATA_MAPS_FOLDER + Y_OUTPUT_NPY_FILENAME, SAVE_TO_DATA_MAPS_FOLDER + MAP_LABELS_FILENAME, using_batches = using_batches)  # Pass in pickle file to load image data from and np save filename
