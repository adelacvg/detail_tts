from pathlib import Path
import torch
import os
import glob
from tqdm import tqdm
# from ttts.utils.utils import get_paths_with_cache
def get_paths_with_cache(search_path, cache_path=None):
    out_paths=None
    if cache_path!=None and os.path.exists(cache_path):
        out_paths = torch.load(cache_path)
    else:
        path = Path(search_path)
        out_paths = find_audio_files(path, ['.wav','.m4a','.mp3'])
        if cache_path is not None:
            print("Building cache..")
            torch.save(out_paths, cache_path)
    return out_paths
def find_audio_files(folder_path, suffixes):
    files = []
    for suffix in suffixes:
        files.extend(glob.glob(os.path.join(folder_path, '**', f'*{suffix}'),recursive=True))
    return files
os.environ["MODELSCOPE_CACHE"] = "./"


def phase1_vad_and_sample(file_paths, out_path, max_workers):
    paths = [[file_path, out_path] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_vad, paths), total=len(file_paths), desc="VAD"))
    results = [result for result in results if result is not None]

def phase2_filter_and_transcript_and_to_jsonl(file_paths, out_path, max_workers):
    paths = [[file_path, out_path] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_file_asr, paths), total=len(file_paths), desc="ASR"))
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    # phase 1
    from vad_process import process_file_vad
    print("---------------phase1-----------------")
    files = get_paths_with_cache('/mnt/nas1/kaixuan/data/Podcast_raw/20240614/')
    out_path = 'datasets/podcast0614'
    Path(out_path).mkdir(exist_ok = True, parents=True)
    phase1_vad_and_sample(files, out_path, 12)

    # phase 2 
    from asr_process import process_file_asr
    print("---------------phase2-----------------")
    files = get_paths_with_cache('datasets/podcast0614')
    out_path = 'datasets/podcast0614.jsonl'
    phase2_filter_and_transcript_and_to_jsonl(files, out_path, 12)


