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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class NumbersAndLettersCNN(pl.LightningModule):
    ''' Implementation of CNN to detect numbers and letters. '''
    def __init__(self, input_dim, output_classes, img_labels):
        super().__init__()
        self.img_labels = img_labels
        self.cross_ent_weight = self.init_cross_entropy_weights()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(256 * 1 * 3, output_classes)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.5)

    def init_cross_entropy_weights(self):
        w = [1 for i in self.img_labels] # init with weight = 1 for each class
        # Increase weight of tricky classes like [0,1,5,G,I,O,S,l,o,r]
        for i in [0,1,5,16,18,24,28,47,50,53]:
            w[i] += 1
        return torch.tensor(w, device='cuda', dtype=torch.float32)

    def forward(self, x):
        ''' Forward pass '''
        x = self.pool(self.dropout1(F.relu(self.conv1(x))))
        x = self.pool(self.dropout2(F.relu(self.conv2(x))))
        x = self.pool(self.dropout2(F.relu(self.conv3(x))))
        x = self.pool(self.dropout2(F.relu(self.conv4(x))))
        x = self.dropout2(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x.float())
        loss = F.cross_entropy(output, y.long(), weight=self.cross_ent_weight)
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
        acc = torch.mean((torch.argmax(output, axis=1) == y).float())
        return {'test_acc': acc,
                'test_pred': torch.argmax(output, axis=1),
                'test_actual': y}

    def test_epoch_end(self, outputs):
        test_acc = torch.squeeze(torch.stack([x['test_acc'] for x in outputs]).float()).mean()
        test_pred = torch.cat([x['test_pred'] for x in outputs]).cpu().numpy()
        test_actual = torch.cat([x['test_actual'] for x in outputs]).cpu().numpy()

        conf_mat = confusion_matrix(test_actual, test_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=self.img_labels)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues,
                         ax=None, xticks_rotation='vertical')
        disp.figure_.set_size_inches(22, 22)

        self.logger.experiment.log({"confusion_matrix":disp.figure_})
        self.logger.log_metrics({"test_acc":test_acc})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

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
        self.img_dataset, self.img_classes, self.img_labels = self.load_data(data_dir)
        self.batch_size = batch_size
        self.nal_train = None
        self.nal_val = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'): # Create all datasets
            # Creating transforms
            transform = transforms.Compose([
                transforms.Resize((90, 120)), # Scale down image
                transforms.Normalize((0.0583), (0.2322)),
                transforms.GaussianBlur(3),
                transforms.RandomRotation(30),
                transforms.Lambda(lambda img: img + np.random.normal(size=np.array(img.shape), scale=0.1)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])

            dataset = NumbersAndLettersDataset(self.img_dataset, self.img_classes, transform)

            # Creating train, val datasets according to an 85-15 split
            self.nal_train, self.nal_val = train_test_split(dataset, test_size=0.15)

    def train_dataloader(self):
        return DataLoader(self.nal_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.nal_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.nal_val, batch_size=self.batch_size, num_workers=4)

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

        # Prepare using Label Encoding
        le = LabelEncoder()
        le.fit(classes)
        labels = le.classes_
        classes = torch.tensor(le.transform(classes))
        return dataset, classes, labels
