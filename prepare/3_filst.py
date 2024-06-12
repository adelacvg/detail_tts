import argparse
import json
import os
import re
import wave
from random import shuffle

from utils import utils
from loguru import logger
from tqdm import tqdm

pattern = re.compile(r'^[\.a-zA-Z0-9_\/]+$')

def get_wav_duration(file_path):
    try:
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频帧数
            n_frames = wav_file.getnframes()
            # 获取采样率
            framerate = wav_file.getframerate()
            # 计算时长（秒）
            return n_frames / float(framerate)
    except Exception as e:
        logger.error(f"Reading {file_path}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", type=str, default="./vqvae/filelists/train.txt", help="path to train list")
    parser.add_argument("--source_dir", type=str, default="./test", help="path to source dir")
    parser.add_argument("--speech_encoder", type=str, default="vec768l12", help="choice a speech encoder|'vec768l12','vec256l9','hubertsoft','whisper-ppg','cnhubertlarge','dphubert','whisper-ppg-large','wavlmbase+'")
    parser.add_argument("--vol_aug", action="store_true", help="Whether to use volume embedding and volume augmentation")
    parser.add_argument("--tiny", action="store_true", help="Whether to train sovits tiny")
    args = parser.parse_args()
    
    train = []
    val = []
    idx = 0
    spk_dict = {}
    spk_id = 0

    for speaker in tqdm(os.listdir(args.source_dir)):
        spk_dict[speaker] = spk_id
        spk_id += 1
        wavs = []

        for file_name in os.listdir(os.path.join(args.source_dir, speaker)):
            if not file_name.endswith("wav"):
                continue
            if file_name.startswith("."):
                continue

            file_path = "/".join([args.source_dir, speaker, file_name])

            if not pattern.match(file_name):
                logger.warning("Detected non-ASCII file name: " + file_path)

            if get_wav_duration(file_path) < 0.3:
                logger.info("Skip too short audio: " + file_path)
                continue

            wavs.append(file_path)

        shuffle(wavs)
        train += wavs[:]

    shuffle(train)

    logger.info("Writing " + args.train_list)
    with open(args.train_list, "w") as f:
        for fname in tqdm(train):
            wavpath = fname
            f.write(wavpath + "\n")
