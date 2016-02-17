import pickle
from sklearn import svm
import cv2
import os
from scipy import ndimage
import numpy as np
import random

img_to_country = pickle.load(open('000_img_to_country', 'r'))
print 'done loading pickle files'
train_files = os.listdir('train')
test_files = os.listdir('test')
hog = cv2.HOGDescriptor()
X = []
y = []
for tf in train_files:
    im = cv2.imread('train/' + tf)
    im = ndimage.rotate(im, 90)
    im = cv2.resize(im, (90,150))
    h = np.squeeze(hog.compute(im))
    X.append(h)
    img = tf[:-4]
    y.append(img_to_country[img])

print 'done loading training data'
clf = svm.SVC()
clf.fit(X,y)
print 'done training'

X = []
y = []
for tf in test_files:
    im = cv2.imread('test/' + tf)
    im = ndimage.rotate(im, 90)
    im = cv2.resize(im, (90,150))
    h = np.squeeze(hog.compute(im))
    X.append(h)
    img = tf[:-4]
    y.append(img_to_country[img])

print 'done loading test data'
y_pred = clf.predict(X)
acc = (y_pred == y).mean()
print 'test accuracy', acc

