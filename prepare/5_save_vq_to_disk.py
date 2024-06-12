import argparse
import functools
import multiprocessing
import os
import json

import torch
import torchaudio
from tqdm import tqdm
from prepare.extract_vq import process_vq

def read_jsonl(path):
    path = os.path.expanduser(path)
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def extract_vq(file_paths, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_vq, file_paths), total=len(file_paths), desc="VQ"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',default='ttts/datasets/data_v4.jsonl')
    args = parser.parse_args()
    paths = read_jsonl(args.json_path)
    paths = [os.path.join('/home/hyc/tortoise_plus_zh',path['path']) for path in paths]
    num_threads = 12
    extract_vq(paths, num_threads)