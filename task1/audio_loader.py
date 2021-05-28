'''
Contains the implementation of the DataLoader for the audio data.
'''
import os
import random
from math import floor, ceil
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
import torchvision
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class AudioCommandsDataset(Dataset):
    ''' Dataset for audio data. '''
    def __init__(self, input_data, target, transform=None):
        self.input_data = input_data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        audio = self.input_data[idx]
        if self.transform: audio = self.transform(audio)
        return (audio, self.target[idx])

class AudioCommandsModule(pl.LightningDataModule):
    ''' DataModule for loading of dataset. '''
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.dataset, self.classes, self.labels = load_data(data_dir)
        self.batch_size = batch_size
        self.audio_train = None
        self.audio_val = None
        self.audio_test = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'): # Create all datasets
            transform = torchvision.transforms.Compose([
                torchaudio.transforms.Resample(16000, 8000),
                torchvision.transforms.Lambda(lambda audio: torch.mul(audio, 10))
            ])
            dataset = AudioCommandsDataset(self.dataset, self.classes, transform)

            # Creating train, val datasets according to an 85-15 split
            self.audio_train, self.audio_val = train_test_split(dataset, test_size=0.1)

            test_data = load_test_data(base_dir)

    def train_dataloader(self):
        return DataLoader(self.audio_train, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.audio_val, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.audio_test, batch_size=self.batch_size, num_workers=8)

def load_core_cls_data(base_dir):
    ''' Load core audio classes data. '''
    dataset, target = [], []
    classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    for cls in classes:
        files = os.listdir(os.path.join(base_dir, cls))
        for f in files:
            audio = torchaudio.load(os.path.join(base_dir, cls, f))[0]
            padding = 16000 - audio.shape[1]
            audio = F.pad(audio, (floor(padding/2), ceil(padding/2)))
            dataset.append(audio)
            target.append(cls)
    return dataset, target

def load_unk_data(base_dir):
    ''' Load a subset of auxiliary classes for the 'unknown' class. '''
    dataset, target = [], []
    auxiliary_classes = ['bed', 'bird', 'cat', 'dog', 'eight', 'five', 'four', 'happy', 'house',
                         'marvin', 'nine', 'one', 'seven', 'sheila', 'six', 'three', 'tree', 'two',
                         'wow', 'zero']
    population = []
    for cls in auxiliary_classes: # Unknown
        files = os.listdir(os.path.join(base_dir, cls))
        population.extend([os.path.join(base_dir, cls, f) for f in files])
    files = random.sample(population, 4000)
    for f in files:
        audio = torchaudio.load(f)[0]
        padding = 16000 - audio.shape[1]
        audio = F.pad(audio, (floor(padding/2), ceil(padding/2)))
        dataset.append(audio)
        target.append('unknown')
    return dataset, target

def load_sil_data(base_dir):
    ''' Load background noise for the 'silence' class. '''
    dataset, target = [], []
    base_dir = os.path.join(base_dir, '_background_noise_')
    while len(dataset) < 2300:
        for f in os.listdir(os.path.join(base_dir)):
            audio_sample = torchaudio.load(os.path.join(base_dir, f),
                                           frame_offset = random.randint(0, 16000*59),
                                           num_frames = 16000)[0]
            dataset.append(audio_sample)
            target.append('silence')
    return dataset, target

def load_data(base_dir):
    ''' Load audio and their classes from disk. '''
    core_dataset, core_target = load_core_cls_data(base_dir)
    unk_dataset, unk_target = load_unk_data(base_dir)
    sil_dataset, sil_target = load_sil_data(base_dir)

    dataset = core_dataset + unk_dataset + sil_dataset
    target = core_target + unk_target + sil_target

    le = LabelEncoder()
    le.fit(target)
    labels = le.classes_
    target = torch.tensor(le.transform(target))
    dataset = torch.stack(dataset)

    return dataset, target, labels
