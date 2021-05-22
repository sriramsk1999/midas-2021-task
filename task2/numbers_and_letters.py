'''
Contains the implementation of the Neural Network and related DataModules
'''
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NumbersAndLettersCNN(pl.LightningModule):
    ''' Implementation of CNN to detect numbers and letters. '''
    def __init__(self, input_dim, output_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 64, 3, padding=2, stride=2)
        self.conv2 = nn.Conv2d(64, 256, 3, padding=2, stride=2)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(256 * 14 * 19, output_classes)

    def forward(self, x):
        ''' Forward pass '''
        x = self.pool(x) # Downsample image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
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
        img = torch.tensor(io.imread(self.input_data[idx]))
        img = img.permute(2, 0, 1) # Reshape to bring channels to first index
        if self.transform:
            pass
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

            dataset = NumbersAndLettersDataset(img_dataset, img_classes)

            # Creating train, test, val datasets according to an 80-10-10 split
            self.nal_train, self.nal_test = train_test_split(dataset, test_size=0.1)
            self.nal_train, self.nal_val = train_test_split(self.nal_train, test_size=0.1)

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
