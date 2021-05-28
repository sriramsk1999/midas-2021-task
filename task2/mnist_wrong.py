'''
Contains the implementation of the MNISTWrong DataModule
'''
import os
from typing import Optional

import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from skimage import io
import pytorch_lightning as pl

class MNISTWrongDataset(Dataset):
    ''' Dataset for wrongly labeled MNIST. '''
    def __init__(self, input_data, target, transform=None):
        self.input_data = input_data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if self.transform: img = self.transform(self.input_data[idx])
        return (img, self.target[idx])

class MNISTWrongModule(pl.LightningDataModule):
    ''' DataModule for loading of dataset. '''
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.img_dataset, self.img_classes = self.load_data("mnistTask")
        self.mnist_wrong_train = None
        self.mnist_wrong_val = None
        self.mnist_wrong_test = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'): # Create all datasets
            # Creating transforms
            transform = transforms.Compose([
                transforms.Resize((90, 120)), # Scale down image
                transforms.Normalize((34.33), (75.80,))
            ])

            dataset = MNISTWrongDataset(self.img_dataset, self.img_classes, transform)
            # Creating train, val datasets according to an 90-10 split
            self.mnist_wrong_train, self.mnist_wrong_val = train_test_split(dataset, test_size=0.1)

            # Transform for actual MNIST
            tensor_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((90, 120)),
                transforms.Normalize((0.1362), (0.2893,))
            ])
            # Test data is from actual MNIST
            self.mnist_wrong_test = datasets.MNIST(os.getcwd(), train=False,
                                             download=True, transform=tensor_transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_wrong_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_wrong_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_wrong_test, batch_size=self.batch_size, num_workers=4)

    def load_data(self, img_dir):
        ''' Load image_paths and their classes from disk. '''
        dataset = []
        classes = []
        for folder in os.listdir(img_dir):
            img_class = int(folder)
            for img in os.listdir(os.path.join(img_dir, folder)):
                img = torch.tensor(io.imread(
                    os.path.join(img_dir, folder, img), as_gray=True), dtype=torch.float32)
                img = torch.unsqueeze(img, 0)
                dataset.append(img)
                classes.append(int(img_class))

        classes = torch.tensor(classes)
        dataset = torch.stack(dataset)
        return dataset, classes
