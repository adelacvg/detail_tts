import os
import random

import numpy as np
import torch
import torch.utils.data

from utils import utils
from modules.mel_processing import spectrogram_torch
from utils.utils import load_filepaths_and_text, load_wav_to_torch
import torchaudio.functional as F
from bpe_tokenizers.voice_tokenizer import VoiceBpeTokenizer
from pypinyin import Style, lazy_pinyin
import torchaudio
import glob
import json

# import h5py

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

"""Multi speaker version"""

def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(folder_path, '**', f'*{suffix}'),recursive=True))
    return files

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        self.tok = VoiceBpeTokenizer('bpe_tokenizers/zh_tokenizer.json')
        self.jsonl_path = hparams.data.training_files
        self.audiopaths_and_text = read_jsonl(self.jsonl_path)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.sampling_rate = hparams.data.sampling_rate
        self.use_sr = hparams.train.use_sr
        self.spec_len = hparams.train.max_speclen

        random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, path_and_text):
        audiopath, text = path_and_text['path'][5:], path_and_text['text']
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
        text = self.tok.encode(text)
        text = torch.LongTensor(text)
        wav, sr = torchaudio.load(audiopath)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = F.resample(wav, sr, self.sampling_rate)
        audio_norm = wav

        # Ideally, all data generated after Mar 25 should have .spec.pt
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                    self.sampling_rate, self.hop_length, self.win_length,
                                    center=False)
        spec = torch.squeeze(spec, 0)

        return spec, audio_norm, text

    def random_slice(self, spec, audio_norm, text):
        l = spec.shape[1]//4*4
        spec = spec[:, :l]
        audio_norm = audio_norm[:, :l * self.hop_length]
        raw_spec = spec
        raw_wav = audio_norm
        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec = spec[:, start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
        l = spec.shape[1]//4*4
        spec = spec[:, :l]
        audio_norm = audio_norm[:, :l * self.hop_length]
        return spec, audio_norm, text, raw_spec, raw_wav

    def __getitem__(self, index):
        try:
            ret = self.random_slice(*self.get_audio(self.audiopaths_and_text[index]))
        except Exception as e:
            print(e)
            return None
        return ret

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])
        max_text_len = max([len(x[2]) for x in batch])+1
        max_raw_spec_len = max([x[3].size(1) for x in batch])
        max_raw_wav_len = max([x[4].size(1) for x in batch])

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        raw_spec_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_raw_spec_len)
        raw_wav_padded = torch.FloatTensor(len(batch), 1, max_raw_wav_len)

        spec_lengths = torch.LongTensor(len(batch))
        text_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        raw_spec_lengths = torch.LongTensor(len(batch))
        raw_wav_lengths = torch.LongTensor(len(batch))

        spec_padded.zero_()
        wav_padded.zero_()
        text_padded.zero_()
        raw_spec_padded.zero_()
        raw_wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] =spec.size(1)

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            text = row[2]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            raw_spec = row[3]
            raw_spec_padded[i, :, :raw_spec.size(1)] = raw_spec
            raw_spec_lengths[i] =raw_spec.size(1)

            raw_wav = row[4]
            raw_wav_padded[i, :, :raw_wav.size(1)] = raw_wav
            raw_wav_lengths[i] = raw_wav.size(1)
        return {
            "spec":spec_padded,
            "spec_length":spec_lengths,
            "raw_spec":raw_spec_padded,
            "raw_spec_length":raw_spec_lengths,
            "wav":wav_padded,
            "wav_length":wav_lengths,
            "raw_wav":raw_wav_padded,
            "raw_wav_length":raw_wav_lengths,
            "text":text_padded,
            "text_length":text_lengths
        }