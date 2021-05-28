'''
Contains the implementation of the Neural Network to detect the audio commands
'''
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class AudioCommandsNN(pl.LightningModule):
    ''' Implementation of CNN to detect audio commands. '''
    def __init__(self, n_input, n_output, labels, stride=16, n_channel=32):
        super().__init__()
        self.labels = labels
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        ''' Forward pass '''
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = torch.flatten(x, 1)
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
        return {'val_acc': acc,
                'val_pred': torch.argmax(output, axis=1),
                'val_actual': y}

    def validation_epoch_end(self, outputs):
        val_acc = torch.squeeze(torch.stack([x['val_acc'] for x in outputs]).float()).mean()
        val_pred = torch.cat([x['val_pred'] for x in outputs]).cpu().numpy()
        val_actual = torch.cat([x['val_actual'] for x in outputs]).cpu().numpy()

        conf_mat = confusion_matrix(val_actual, val_pred, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=self.labels)
        disp = disp.plot(include_values=True, cmap=plt.cm.Blues,
                         ax=None, xticks_rotation='vertical')
        disp.figure_.set_size_inches(22, 22)

        self.logger.experiment.log({"confusion_matrix":disp.figure_})
        self.logger.log_metrics({"val_acc":val_acc})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
