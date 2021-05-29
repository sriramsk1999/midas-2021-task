import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from audio_loader import AudioCommandsModule
from audio_commands import AudioCommandsNN

SAVE_PATH = "models/"
BASE_DIR = 'train/audio'
MODEL_NAME = 'm5_less-unk-sil_notrans'
BATCH_SIZE = 32
TRAIN = True

data_module = AudioCommandsModule(BASE_DIR, BATCH_SIZE)
model = AudioCommandsNN(1, 12, data_module.labels)

# Log metrics to WandB
wandb_logger = pl.loggers.WandbLogger(save_dir='logs/',
                                        name=MODEL_NAME,
                                        project='midas-task-1')
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3
)

trainer = pl.Trainer(gpus=1, logger=wandb_logger,
                    callbacks=[early_stopping])
trainer.fit(model, data_module)

# Save model
torch.save(model.state_dict(), os.path.join(SAVE_PATH, MODEL_NAME))
