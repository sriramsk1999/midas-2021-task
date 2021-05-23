'''
Contains the implementation of the Neural Network and related DataModules
'''
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from skimage import io, util
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NumbersAndLettersCNN(pl.LightningModule):
    ''' Implementation of CNN to detect numbers and letters. '''
    def __init__(self, input_dim, output_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 8, output_classes)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        ''' Forward pass '''
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.pool(self.dropout2(F.relu(self.conv2(x))))
        x = self.pool(self.dropout2(F.relu(self.conv3(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x.float())
        loss = F.cross_entropy(output, y.long())
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x.float())
        loss = F.cross_entropy(output, y.long())
        acc = torch.mean((torch.argmax(output, axis=1) == y).float())
        self.log_dict({'val_loss': loss, 'val_acc': acc}, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x.float())
        loss = F.cross_entropy(output, y.long())
        acc = torch.mean((torch.argmax(output, axis=1) == y).float())
        self.log_dict({'test_loss': loss, 'test_acc': acc}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class NumbersAndLettersDataset(Dataset):
    ''' Dataset for numbers and letters. '''
    def __init__(self, input_data, target, transform=None):
        self.input_data = input_data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        img = io.imread(self.input_data[idx], as_gray=True)
        img = torch.tensor(util.invert(img)) # Invert colours to be white on black
        img = torch.unsqueeze(img, 0)
        if self.transform: img = self.transform(img)
        return (img, self.target[idx])

class NumbersAndLettersModule(pl.LightningDataModule):
    ''' DataModule for loading of dataset. '''
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.nal_train = None
        self.nal_test = None
        self.nal_val = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'): # Create all datasets
            img_dataset, img_classes = self.load_data(self.data_dir)
            print("Data loaded from disk")

            # Prepare target using Label Encoding
            le = LabelEncoder()
            le.fit(img_classes)
            img_classes = torch.tensor(le.transform(img_classes))

            # Creating transforms
            transform = transforms.Compose([
                transforms.Resize((30, 40)), # Scale down image
                transforms.Normalize((0.0583), (0.2322)),
                transforms.GaussianBlur(3),
                transforms.RandomRotation(30),
                transforms.Lambda(lambda img: img + np.random.normal(size=np.array(img.shape), scale=0.05)),
            ])

            dataset = NumbersAndLettersDataset(img_dataset, img_classes, transform)

            # Creating train, test, val datasets according to an ~90-5-5 split
            self.nal_train, self.nal_test = train_test_split(dataset, test_size=0.05)
            self.nal_train, self.nal_val = train_test_split(self.nal_train, test_size=0.05)

    def train_dataloader(self):
        return DataLoader(self.nal_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.nal_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.nal_test, batch_size=self.batch_size, num_workers=4)

    def load_data(self, img_dir):
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
