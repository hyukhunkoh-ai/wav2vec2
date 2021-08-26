import torch
import os
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from wav2vec2 import Wav2Vec2Model
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# epochs = 100000
epochs = 1
bs = 5 # 5 * 4 = 20

#train

# get filename list
train_name_list = []
with os.scandir('./train_data') as files:
    for file in files:
        if file.is_file():
            if file.name.endswith('.wav'):
                train_name_list.append(file.name)
per_bs = int(len(train_name_list) // bs) + 1

# get xdata
train_data = []
for filename in train_name_list:
    audio, sr = sf.read(f'./train_data/{filename}',dtype="float32")
    if len(audio) > 32000:
        train_data.append(list(audio))

input_padding = Wav2Vec2FeatureExtractor.from_pretrained("patrickvonplaten/wav2vec2-base")
train_x = input_padding(train_data,sampling_rate=sr,padding='max_length',padding_value=0,return_tensors="pt",max_length=250000,truncation=True).input_values

train_ds = torch.utils.data.TensorDataset(train_x)
# train_dl = torch.utils.data.DataLoader(train_ds, batch_size=100, shuffle=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)

model = Wav2Vec2Model()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.0005,steps_per_epoch=per_bs, epochs=epochs, pct_start=0.08)


for epoch in range(epochs):
    total_loss = []
    for x in train_dl:
        dx = x[0]
        dx = dx.to(device)
        model.to(device)
        optim.zero_grad()
        loss = model.calculate_loss(dx)
        total_loss.append(loss.item())
        loss.backward()
        optim.step()
        scheduler.step()
    print(np.mean(total_loss))


