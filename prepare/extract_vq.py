import torchaudio
from prepare.load_infer import load_model
import torch
import torchaudio.functional as F
from tqdm import tqdm
import os
import json
from vqvae.utils.data_utils import spectrogram_torch,HParams
import random

model_path = '/home/hyc/detail_tts/vqvae/logs/2024-06-19-07-19-25/model-194.pt'
# device_ids = [4,5,6,7]
# rank = random.choice(device_ids)
# device = f'cuda:{rank}'
device = 'cuda'
vqvae = load_model('vqvae', model_path, 'vqvae/configs/config.json', device)
hps = HParams(**json.load(open('vqvae/configs/config.json')))
def process_vq(path):
    wav_path = path
    try:
        wav,sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = wav.to(device)
        wav44k = F.resample(wav, sr, 44100)
        wav44k = wav44k[:,:int(hps.data.hop_length * 2 * (wav44k.shape[-1]//hps.data.hop_length//2))]
        wav = torch.clamp(wav44k, min=-1.0, max=1.0)
    except Exception as e:
        print(path)
        print(e)
        return
    try:
        with torch.no_grad():
            spec = spectrogram_torch(wav, hps.data.filter_length,
                    hps.data.hop_length, hps.data.win_length, center=False).to(device)
            spec_lengths = torch.LongTensor([spec.shape[-1]]).to(device)
            code = vqvae.encode(spec,spec_lengths).squeeze(0).squeeze(0)
    except Exception as e:
        print(path)
        print(e)
        return
    torch.save(spec.cpu().detach(), path+'.spec.pth')
    outp = path+'.vq.pth'
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    torch.save(code.tolist(), outp)
    return