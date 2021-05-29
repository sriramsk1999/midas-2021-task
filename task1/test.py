import os
from math import floor, ceil
import torch
import torchaudio
import torchvision
import torch.nn.functional as F
from audio_loader import AudioCommandsModule
from audio_commands import AudioCommandsNN

SAVE_PATH = "models/"
LOAD_MODEL_NAME = 'm5_less-unk-sil_notrans'
BASE_DIR = 'train/audio'
TEST_DIR = 'test/audio'
BATCH_SIZE = 32

data_module = AudioCommandsModule(BASE_DIR, BATCH_SIZE)
model = AudioCommandsNN(1, 12, data_module.labels)

model.load_state_dict(torch.load(os.path.join(SAVE_PATH, LOAD_MODEL_NAME),
                                    map_location=torch.device('cuda')))
model.eval()
files = os.listdir(TEST_DIR)
csv = open('submission.csv','w')
csv.write('fname,label\n')
transform = torchvision.transforms.Compose([
    torchaudio.transforms.Resample(16000, 8000),
    torchvision.transforms.Lambda(lambda audio: torch.mul(audio, 10))
])
for idx, f in enumerate(files):
    audio = torchaudio.load(os.path.join(TEST_DIR, f))[0]
    padding = 16000 - audio.shape[1]
    audio = F.pad(audio, (floor(padding/2), ceil(padding/2)))
    # audio = transform(audio)
    audio = torch.unsqueeze(audio, dim=0)
    output = model(audio)
    pred = data_module.labels[torch.argmax(output, axis=1)]
    print('%s/%s %s,%s' %(idx, len(files), f, pred), end='\r')
    csv.write('%s,%s\n' %(f, pred))
csv.close()
