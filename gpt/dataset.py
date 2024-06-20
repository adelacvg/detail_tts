import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio
from pypinyin import Style, lazy_pinyin

from bpe_tokenizers.voice_tokenizer import VoiceBpeTokenizer
import json
import os
from pathlib import Path

def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def write_jsonl(path, all_paths):
    with open(path,'w', encoding='utf-8') as file:
        for item in all_paths:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

class GptTtsDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.tok = VoiceBpeTokenizer('bpe_tokenizers/zh_tokenizer.json')
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)
        self.audiopaths_and_text = sorted(self.audiopaths_and_text,
                key=lambda x: x['path'])
        self.sampling_rate = opt['dataset']['sampling_rate']
    def get_text_and_wav(self, audiopath_and_text):
        audiopath, text = audiopath_and_text['path'][5:], audiopath_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
        text = self.tok.encode(text)
        text = LongTensor(text)
        # Fetch quantized MELs
        wav, sr = torchaudio.load(audiopath)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        return text, wav
    def is_same_spk(self, path1, path2):
        name1 = Path(path1).parent.name
        name2 = Path(path2).parent.name
        if name1 == name2:
            return True
        return False
    def __getitem__(self, index):
        try:
            # Fetch text and add start/stop tokens.
            audiopath_and_text = self.audiopaths_and_text[index]
            text, wav = self.get_text_and_wav(audiopath_and_text)
            pths = self.audiopaths_and_text
            wav_length = wav.shape[-1]
        except Exception as e:
            print(e)
            return None
        if text.shape[-1]>=1600:
            return None 
        # load wav
        return text, wav, wav_length

    def __len__(self):
        return len(self.audiopaths_and_text)


class GptTtsCollater():

    def __init__(self,cfg):
        self.cfg=cfg
    def __call__(self, batch):
        batch = [x for x in batch if x is not None]
        if len(batch)==0:
            return None
        text_lens = [len(x[0]) for x in batch]
        max_text_len = max(text_lens)+1
        wav_lens = [x[1].shape[-1] for x in batch]
        max_wav_len = max(wav_lens)
        wav_lens = [x[2] for x in batch]
        max_wav_len = max(wav_lens)
        texts = []
        wavs = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text, wav, wav_length = b
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            wavs.append(F.pad(wav, (0, max_wav_len-wav.shape[-1]), value=0))

        padded_texts = torch.stack(texts)
        padded_wavs = torch.stack(wavs)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_wav': padded_wavs,
            'wav_lens': LongTensor(wav_lens),
        }


if __name__ == '__main__':
    params = {
        'mode': 'gpt_tts',
        'path': 'E:\\audio\\LJSpeech-1.1\\ljs_audio_text_train_filelist.txt',
        'phase': 'train',
        'n_workers': 0,
        'batch_size': 16,
        'mel_vocab_size': 512,
    }
    cfg = json.load(open('ttts/gpt/config.json'))
    ds = GptTtsDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, **cfg['dataloader'], collate_fn=GptTtsCollater(cfg))
    i = 0
    m = []
    max_text = 0
    max_mel = 0
    for b in tqdm(dl):
        break
