'''
Train and test the neural network
'''

import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from numbers_and_letters import NumbersAndLettersCNN, NumbersAndLettersModule
from mnist import MNISTModule

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='numbers_and_letters')
parser.add_argument('--pretrained', default=False)
args = parser.parse_args()

SAVE_PATH = "models/"
MODEL_NAME = '5conv1fc_mnist'
LOAD_MODEL_NAME = '5conv1fc_numbers'
BATCH_SIZE = 32
NUMBERS_ONLY = True

if args.dataset == 'numbers_and_letters':
    BASE_DIR = "train"
    INPUT_DIM = torch.tensor([3, 900, 1200])

    # Create DataModule to handle loading of dataset
    data_module = NumbersAndLettersModule(BASE_DIR, BATCH_SIZE, NUMBERS_ONLY)
    model = NumbersAndLettersCNN(INPUT_DIM, len(data_module.img_labels),
                                 data_module.img_labels, NUMBERS_ONLY)
elif args.dataset == 'mnist':
    data_module = MNISTModule(BATCH_SIZE)
    INPUT_DIM = torch.tensor([1, 28, 28])
    model = NumbersAndLettersCNN(INPUT_DIM, 10, ['0','1','2','3','4',
                                                 '5','6','7','8','9'], NUMBERS_ONLY)
else:
    print("Invalid dataset choice")
    exit(0)

# Log metrics to WandB
wandb_logger = pl.loggers.WandbLogger(save_dir='logs/',
                                        name=MODEL_NAME,
                                        project='midas-task-2')
early_stopping = EarlyStopping(
    monitor='val_loss',
)

if args.pretrained:
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, LOAD_MODEL_NAME),
                                     map_location=torch.device('cuda')))

trainer = pl.Trainer(gpus=1, logger=wandb_logger,
                     callbacks=[early_stopping])
trainer.fit(model, data_module)
trainer.test(model=model, datamodule=data_module)

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_PATH, MODEL_NAME))
