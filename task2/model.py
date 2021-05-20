'''
Contains the implementation of the neural networks and related Datasets.
'''
import torch
from skimage import io
from torch.utils.data import Dataset

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
        return (img, self.target[idx])
