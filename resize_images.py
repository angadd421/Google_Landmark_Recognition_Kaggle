import os
from tqdm import tqdm # progess bar
from PIL import Image
from io import BytesIO

path = '/share/jproject/fg538/r-006-gpu-4/data'

for folder in os.listdir('{}/train/'.format(path))[71:]:
    for file in tqdm(os.listdir('{}/train/{}/'.format(path, folder))):
        filename = '{}/train_resized/{}/{}'.format(path, folder, file)
        
        if os.path.exists(filename):
            continue
            
        img = Image.open('{}/train/{}/{}'.format(path, folder, file))
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        img.resize((227, 227)).save(filename)
