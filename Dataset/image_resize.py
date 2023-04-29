"""
In order to build the dataset we take original images organized in directories named with the name of the actor,
extract the first 12 images, resize them to (224,224) and assign each of them to the TRAIN set.
The same is done for the other 8 images, assigned to the TEST set.
"""
from PIL import Image
import os

actors = os.listdir('./Originali')

for c, name in enumerate(actors):
	for i in range(1, 13):
	  src = './Originali/{}/{}.jpg'.format(name,i)
	  src_new = './TRAIN/{}/{}_cropped.jpg'.format(name,i)
	  image = Image.open(src)
	  image = image.convert('RGB')
	  image_resized = image.resize((224, 224))
	  image_resized.save(src_new)

actors = os.listdir('./Originali')

for c, name in enumerate(actors):
	for i in range(13, 21):
	  src = './Originali/{}/{}.jpg'.format(name,i)
	  src_new = './TEST/{}/{}_cropped.jpg'.format(name,i)
	  image = Image.open(src)
	  image = image.convert('RGB')
	  image_resized = image.resize((224, 224))
	  image_resized.save(src_new)
