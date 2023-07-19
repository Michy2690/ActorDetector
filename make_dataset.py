from PIL import Image
import os
DATASET_PATH = "./Dataset_nuovo"
SRC = "./actor_faces"
actors = os.listdir(SRC)

for c, name in enumerate(actors):
        print(name, c)
        os.mkdir('{}/{}'.format(DATASET_PATH,name))
        images = os.listdir('{}/{}/'.format(SRC, name))
        for i, img_name in enumerate(images):
          if i == 20:
               break
          src = './{}/{}/{}'.format(SRC,name,img_name)
          src_new = './{}/{}/{}'.format(DATASET_PATH,name,img_name)
          image = Image.open(src)
          image = image.convert('RGB')
          image_resized = image.resize((224, 224))
          image_resized.save(src_new)

SRC = "./actress_faces"
actors = os.listdir(SRC)

for c, name in enumerate(actors):
        print(name, c)
        os.mkdir('{}/{}'.format(DATASET_PATH,name))
        images = os.listdir('{}/{}/'.format(SRC, name))
        for i, img_name in enumerate(images):
          if i == 20:
               break
          src = './{}/{}/{}'.format(SRC,name,img_name)
          src_new = './{}/{}/{}'.format(DATASET_PATH,name,img_name)
          image = Image.open(src)
          image = image.convert('RGB')
          image_resized = image.resize((224, 224))
          image_resized.save(src_new)