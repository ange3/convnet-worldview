import os
from PIL import Image
import random
from sets import Set
import pickle

"""
labels = open('mediaeval2015_placing_locale_train', 'r')
img_to_row = {}
img_to_country = {}
countries = Set([])
row = 0
for columns in (raw.strip().split() for raw in labels):
	img_to_row[columns[0]] = row
	country = columns[4]
	cutoff = columns[4].find('@')
	if cutoff > 0:
		country = columns[4][0:cutoff]
	countries.add(country)
	img_to_country[columns[0]] = country
	row = row + 1
	#print country
print len(countries)

pickle.dump(img_to_country, open('img_to_country', 'wb'))
pickle.dump(img_to_row, open('img_to_row', 'wb'))
"""

img_to_country = pickle.load(open('img_to_country', 'r'))

subset_img_to_country = {}

country_subset = Set(['Canada', 'Germany', 'Spain', 'France', 'Italy'])
files = os.listdir('.')
for f in files:
	# assume that image files start with digits, and all other files do not
	if f[0].isdigit() and f in img_to_country:
		cf = img_to_country[f]
		if cf in country_subset:
			subset_img_to_country[f] = cf

			"""
			img = Image.open(f)
			#print img.size[0], img.size[1]
			if img.size[1] > img.size[0]:
				img = img.rotate(90)
			img = img.resize((500,300))
			prefix = 'train/'
			rand = random.randint(1,11)
			if rand == 10:
				prefix = 'validation/'
			if rand == 11:
				prefix = 'test/'
			img.save(prefix + f + '.png')
			"""
pickle.dump(subset_img_to_country, open('000_img_to_country', 'wb'))
		
		
