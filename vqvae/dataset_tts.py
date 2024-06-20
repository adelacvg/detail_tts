import os
import random

import numpy as np
import torch
import torch.utils.data

from utils import utils
from modules.mel_processing import spectrogram_torch
from utils.utils import load_filepaths_and_text, load_wav_to_torch
import torchaudio.functional as F
import torchaudio
import glob

# import h5py


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
        self.audiopaths = find_audio_files(audiopaths, ['.wav'])
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
        random.shuffle(self.audiopaths)
        
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        wav, sr = torchaudio.load(filename)
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        wav = F.resample(wav, sr, self.sampling_rate)
        audio_norm = wav
        spec_filename = filename.replace(".wav", ".spec.pt")

        # Ideally, all data generated after Mar 25 should have .spec.pt
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                    self.sampling_rate, self.hop_length, self.win_length,
                                    center=False)
        spec = torch.squeeze(spec, 0)

        return spec, audio_norm

    def random_slice(self, spec, audio_norm):
        if spec.shape[1] > 800:
            start = random.randint(0, spec.shape[1]-800)
            end = start + 790
            spec = spec[:, start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]
        l = spec.shape[1]//4*4
        spec = spec[:, :l]
        audio_norm = audio_norm[:, :l * self.hop_length]
        return spec, audio_norm

    def __getitem__(self, index):
        try:
            ret = self.random_slice(*self.get_audio(self.audiopaths[index]))
        except:
            return None
        return ret

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[1] for x in batch]),
            dim=0, descending=True)

        max_spec_len = max([x[0].size(1) for x in batch])
        max_wav_len = max([x[1].size(1) for x in batch])

        spec_padded = torch.FloatTensor(len(batch), batch[0][0].shape[0], max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        lengths = torch.LongTensor(len(batch))

        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[0]
            spec_padded[i, :, :spec.size(1)] = spec
            lengths[i] =spec.size(1)

            wav = row[1]
            wav_padded[i, :, :wav.size(1)] = wav
        return (spec_padded, wav_padded,lengths)