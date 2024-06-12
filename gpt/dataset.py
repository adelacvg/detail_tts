import os
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import LongTensor
from tqdm import tqdm
import torchaudio
from pypinyin import Style, lazy_pinyin

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
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
        self.tok = VoiceBpeTokenizer('ttts/gpt/gpt_tts_tokenizer.json')
        self.jsonl_path = opt['dataset']['path']
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)
        self.audiopaths_and_text = sorted(self.audiopaths_and_text,
                key=lambda x: x['path'])
    def get_text_and_vq(self, audiopath_and_text):
        audiopath, text = audiopath_and_text['path'], audiopath_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
        text = self.tok.encode(text)
        text = LongTensor(text)
        # Fetch quantized MELs
        quant_path = audiopath + '.dur.pth'
        vq = LongTensor(torch.load(quant_path))
        return text, vq
    def get_wav_len(self, path):
        wav,sr = torchaudio.load(audiopath)
        wav = torchaudio.functional.resample(wav, sr, 32000)
        wav_length = wav.shape[-1]
        return wav_length
    def is_same_spk(self, path1, path2):
        name1 = Path(path1).parent.name
        name2 = Path(path2).parent.name
        if name1[:3]=='SSB' and name2[:3]=='SSB':
            if name1[:7]==name2[:7]:
                return True
        elif name1[:2]=='vo' and name2[:2]=='vo':
            if name1.split('_')[-2] == name2.split('_')[-2] or \
            name1.split('_')[-3] == name2.split('_')[-3]:
                return True
        else:
            if name1 == name2:
                return True
        return False
    def __getitem__(self, index):
        try:
            # Fetch text and add start/stop tokens.
            audiopath_and_text = self.audiopaths_and_text[index]
            text, vq = self.get_text_and_vq(audiopath_and_text)
            pths = self.audiopaths_and_text
            l = self.__len__()
            text_prepre, vq_prepre, prepre_wav_len = None,None,None
            text_postpost, vq_postpost, postpost_wav_len = None,None,None
            text_pre, vq_pre, pre_wav_len = None, None, None
            text_post, vq_post, post_wav_len = None, None, None
            prepre_ind = random.randint(2, 9)
            if self.is_same_spk(pths[(index-prepre_ind+l)%l]['path'], audiopath_and_text['path']):
                text_prepre, vq_prepre = self.get_text_and_vq(pths[(index-prepre_ind+l)%l])
                vq_prepre = vq_prepre[1:-1]
                prepre_wav_len = vq_prepre.shape[-1]*1280
            if self.is_same_spk(pths[(index-1+l)%l]['path'], audiopath_and_text['path']):
                text_pre, vq_pre = self.get_text_and_vq(pths[(index-1+l)%l])
                vq_pre = vq_pre[1:-1]
                pre_wav_len = vq_pre.shape[-1]*1280
            if self.is_same_spk(pths[(index+1+l)%l]['path'], audiopath_and_text['path']):
                text_post, vq_post = self.get_text_and_vq(pths[(index+1+l)%l])
                vq_post = vq_post[1:-1]
                post_wav_len = vq_post.shape[-1]*1280
            postpost_ind = random.randint(2, 9)
            if self.is_same_spk(pths[(index+postpost_ind+l)%l]['path'], audiopath_and_text['path']):
                text_postpost, vq_postpost = self.get_text_and_vq(pths[(index+postpost_ind+l)%l])
                vq_postpost = vq_postpost[1:-1]
                postpost_wav_len = vq_postpost.shape[-1]*1280
            wav_length = vq.shape[-1]*1280
            if text_pre is not None:
                text = torch.cat((text_pre,text))
                vq = torch.cat((vq_pre, vq))
                wav_length = wav_length + pre_wav_len
            if text_prepre is not None:
                text = torch.cat((text_prepre,text))
                vq = torch.cat((vq_prepre, vq))
                wav_length = wav_length + prepre_wav_len
            if text_post is not None:
                text = torch.cat((text,text_post))
                vq = torch.cat((vq, vq_post))
                wav_length = wav_length + post_wav_len
            if text_postpost is not None:
                text = torch.cat((text,text_postpost))
                vq = torch.cat((vq, vq_postpost))
                wav_length = wav_length + postpost_wav_len
        except Exception as e:
            print(e)
            return None
        if text.shape[-1]>=800 or vq.shape[-1]>=1600:
            return None 
        # load wav
        return text, vq, wav_length

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
        texts = []
        qmels = []
        wavs = []
        # This is the sequential "background" tokens that are used as padding for text tokens, as specified in the DALLE paper.
        for b in batch:
            text, qmel, wav_length = b
            text = F.pad(text, (0, max_text_len-len(text)), value=0)
            texts.append(text)
            qmels.append(F.pad(qmel, (0, max_qmel_len-len(qmel)), value=0))

        padded_qmel = torch.stack(qmels)
        padded_texts = torch.stack(texts)
        return {
            'padded_text': padded_texts,
            'text_lengths': LongTensor(text_lens),
            'padded_qmel': padded_qmel,
            'qmel_lengths': LongTensor(qmel_lens),
            'wav_lens': LongTensor(wav_lens)
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
