import os
from tqdm import tqdm # progess bar
from PIL import Image
import numpy as np
x = []
y = []

path = '/share/jproject/fg538/r-006-gpu-4/data'

for folder in os.listdir('{}/train/'.format(path)):
    for file in tqdm(os.listdir('{}/train/{}/'.format(path, folder))):
        filename = '{}/train_resized/{}/{}'.format(path, folder, file)
        im = Image.open(filename)
        arr = np.array(im)
        x.append(arr)
        y.append(folder)
        im.close()
        
        
np.savez_compressed('train-images', x)
np.savez_compressed('train-labels', y)
