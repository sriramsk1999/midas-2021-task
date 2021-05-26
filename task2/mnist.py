'''
Contains the implementation of the MNIST DataModule
'''
import os
from typing import Optional

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

class MNISTModule(pl.LightningDataModule):
    ''' DataModule for loading of dataset. '''
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'): # Create all datasets
            # Creating transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((90, 120)), # Scale down image
                transforms.Normalize((0.1362,), (0.2893,))
            ])

            self.mnist_train = datasets.MNIST(os.getcwd(), train=True,
                                              download=True, transform=transform)
            self.mnist_test = datasets.MNIST(os.getcwd(), train=False,
                                             download=True, transform=transform)
            self.mnist_train, self.mnist_val = train_test_split(self.mnist_train, test_size=0.1)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
