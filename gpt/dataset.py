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
    def get_text_and_vq(self, audiopath_and_text):
        audiopath, text = audiopath_and_text['path'][5:], audiopath_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
        text = self.tok.encode(text)
        text = LongTensor(text)
        # Fetch quantized MELs
        quant_path = audiopath + '.vq.pth'
        vq = LongTensor(torch.load(quant_path))
        spec_path = audiopath + '.spec.pth'
        spec = torch.load(spec_path).detach().squeeze(0)
        return text, vq, spec
    def is_same_spk(self, path1, path2):
        name1 = Path(path1).parent.name
        name2 = Path(path2).parent.name
        if name1 == name2:
            return True
        return False
    def __getitem__(self, index):
        squeeze_scale = 2048
        try:
            # Fetch text and add start/stop tokens.
            audiopath_and_text = self.audiopaths_and_text[index]
            text, vq, spec = self.get_text_and_vq(audiopath_and_text)
            pths = self.audiopaths_and_text
            wav_length = vq.shape[-1]*squeeze_scale
        except Exception as e:
            print(e)
            return None
        if text.shape[-1]>=1600 or vq.shape[-1]>=3200:
            return None 
        # load wav
        return text, vq, wav_length, spec

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
        qmel_lens = [len(x[1]) for x in batch]
        max_qmel_len = max(qmel_lens)+1
        wav_lens = [x[2] for x in batch]
        max_wav_len = max(wav_lens)
        spec_lens = [x[3].shape[1] for x in batch]
        max_spec_len = max(spec_lens)+1
        texts = []
        qmels = []
        wavs = []
        specs = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text, qmel, wav_length, spec = b
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            qmels.append(F.pad(qmel, (0, max_qmel_len-len(qmel)), value=0))
            specs.append(F.pad(spec,(0, max_spec_len-spec.shape[1]), value=0))

        padded_qmel = torch.stack(qmels)
        padded_texts = torch.stack(texts)
        padded_spec = torch.stack(specs)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_qmel': padded_qmel,
            'qmel_lengths': LongTensor(qmel_lens),
            'wav_lens': LongTensor(wav_lens),
            'padded_spec': padded_spec,
            'spec_lengths': LongTensor(spec_lens)
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
