from os import listdir
from os.path import isfile, join
import pickle, csv

PATH_TO_PLACE_TASK_DATA = '../../mediaeval_placing_task_2015_data/'
PATH_TO_PLACE_TASK_DATA_TRAIN = '../data/train'
PATH_TO_PLACE_TASK_DATA_TEST = '../data/test'
PATH_TO_PLACE_TASK_DATA_VAL = '../data/validation'

TRUTH_LABELS_FILENAME = 'mediaeval2015_placing_locale_train'


def get_image_filenames(path):
  '''
  Returns a list of filenames in the folder of the specified path
  '''
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
    for line in csv.reader(tsv, dialect="excel-tab"):
      if ncount % 50000 == 0:
        print 'processed {} * 50k photos'.format(ncount/50000)
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
    obj_id = '00cb928aff9719f0d57c21e371288b'
    print 'obj_id = {}'.format(obj_id)
    print 'original value = {}'.format(image_info_map[obj_id])
    with open(pickle_filename, 'r') as f:
      test_map = pickle.load(f)
      print 'pickled value = {}'.format(test_map[obj_id])

  return image_info_map


def map_image_id_to_node(pickle_file, image_node_map, image_filenames_list, truth_labels_all):
  '''
  Loads the pickle file and updates the given map with the following key, value pairs.
    Key: names of images from the given list
    Value: associated node location names taken from the truth labels
    {image ID: (node_id, node_name)}
  '''
  truth_labels_for_image_filenames = {}

  for image_id in image_filenames_list:
    # node_id = 
    # node_name = 
    # image_node_map[image_id] = (node_id, node_name)
    pass


if __name__ == "__main__":

  # test pickle file
  # obj_id = '001188ddf4be8e85477ca09d6b742f'
  # print 'obj_id = {}'.format(obj_id)
  # # print 'original value = {}'.format(image_info_map[obj_id])
  # with open('../data_maps/image_id_to_node_info_000_labels.pickle', 'r') as f:
  #   test_map = pickle.load(f)
  #   print 'pickled value = {}'.format(test_map[obj_id])

  # path = PATH_TO_PLACE_TASK_DATA + '000/'
  image_filenames_train = get_image_filenames(PATH_TO_PLACE_TASK_DATA_TRAIN)
  image_filenames_test = get_image_filenames(PATH_TO_PLACE_TASK_DATA_TEST)
  image_filenames_val = get_image_filenames(PATH_TO_PLACE_TASK_DATA_VAL)
  image_filenames = image_filenames_train + image_filenames_test + image_filenames_val

  # truth_labels_train = open_labels_data(pickle_filename = '../data_maps/image_id_to_node_info_train_labels.pickle')
  truth_labels_000 = open_labels_data(image_filenames, pickle_filename = '../data_maps/image_id_to_node_info_000_labels.pickle')

  # truth_labels_for_image_filenames = map_image_id_to_node(image_node_map, image_filenames, truth_labels_all)