'''
Input pipeline, training and testing setup
'''
import os

import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import NumbersAndLettersDataset

def make_dataset(img_dir):
    ''' Load image_paths and their classes from disk. '''
    dataset = []
    classes = []
    for folder in os.listdir(img_dir):
        img_class = int(folder[-2:]) # Extract last 2 digits of folder name
        if img_class < 11:
            img_class = str(img_class - 1) # 0-9
        elif img_class < 37:
            img_class = chr(img_class + 54) # A-Z
        else: img_class = chr(img_class + 60) # a-z
        for img in os.listdir(os.path.join(img_dir, folder)):
            img_path = os.path.join(img_dir, folder, img)
            dataset.append(img_path)
            classes.append(img_class)
    return dataset, classes

def format_and_load(dataset, target, batchsize):
    '''
    Formats the input dataset as expected by the network and returns a DataLoader.
    '''
    target = torch.tensor(target)

    loaded_data = DataLoader(NumbersAndLettersDataset(dataset, target),
                             batch_size=batchsize, num_workers=8)
    return loaded_data

SEED = 42 # Set a global seed for reproducible results
BATCH_SIZE = 64

pl.utilities.seed.seed_everything(SEED)

##################
# INPUT PIPELINE #
##################

BASE_DIR = "train"
img_dataset, img_classes = make_dataset(BASE_DIR)
print("Data loaded from disk")

# Prepare target using Label Encoding
le = LabelEncoder()
le.fit(img_classes)
img_classes = le.transform(img_classes)

# Split dataset for train/test
train_x, test_x, train_y, test_y = train_test_split(img_dataset, img_classes,
                                                    test_size=0.2, random_state=SEED)

# Set up DataLoaders
train_loader = format_and_load(train_x, train_y, BATCH_SIZE)
test_loader = format_and_load(test_x, test_y, BATCH_SIZE)
print("Constructed Dataloader")
