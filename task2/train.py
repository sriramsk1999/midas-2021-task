'''
Train and test the neural network
'''

import os

import torch
import pytorch_lightning as pl

from numbers_and_letters import NumbersAndLettersCNN, NumbersAndLettersModule

SEED = 42 # Set a global seed for reproducible results
BATCH_SIZE = 4
BASE_DIR = "train"
SAVE_PATH = "models/"
MODEL_NAME = 'numbers-and-letters-cnn'

INPUT_DIM = torch.tensor([3, 900, 1200])
OUTPUT_CLASSES = 62

pl.utilities.seed.seed_everything(SEED)

# Create DataModule to handle loading of dataset
data_module = NumbersAndLettersModule(BASE_DIR, BATCH_SIZE)

# Train and test model

model = NumbersAndLettersCNN(INPUT_DIM, OUTPUT_CLASSES)

# Log metrics to WandB
wandb_logger = pl.loggers.WandbLogger(save_dir='logs/',
                                        name=MODEL_NAME,
                                        project='midas-task-2')
trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=1)
trainer.fit(model, data_module)
trainer.test(model=model, datamodule=data_module)

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_PATH, MODEL_NAME))
