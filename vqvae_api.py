from prepare.load_infer import load_model
import torch
import torchaudio.functional as F
from tqdm import tqdm
import os
import json
from vqvae.utils.data_utils import spectrogram_torch,HParams
import random
import torchaudio

model_path = '/home/hyc/detail_tts/vqvae/logs/2024-06-19-12-53-53/model-205.pt'
device = 'cuda'
vqvae = load_model('vqvae', model_path, 'vqvae/configs/config.json', device)
hps = HParams(**json.load(open('vqvae/configs/config.json')))
wav_path = '0.wav'
wav,sr = torchaudio.load(wav_path)
if wav.shape[0] > 1:
    wav = wav[0].unsqueeze(0)
wav = wav.to(device)
wav44k = F.resample(wav, sr, 44100)
wav44k = wav44k[:,:int(hps.data.hop_length *4* (wav44k.shape[-1]//hps.data.hop_length//4))]
wav = torch.clamp(wav44k, min=-1.0, max=1.0)
wav_length = torch.LongTensor([wav.shape[1]])
spec = spectrogram_torch(wav, hps.data.filter_length,
                    hps.data.hop_length, hps.data.win_length, center=False)
spec_length = torch.LongTensor([
    x//hps.data.hop_length for x in wav_length]).to(device)
with torch.no_grad():
    wav,_ = vqvae.infer(spec, spec_length, spec, spec_length)
torchaudio.save('gen.wav', wav.squeeze(0).cpu(), 44100)