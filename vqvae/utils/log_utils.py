import logging
import os
import functools
import math
from pathlib import Path
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import glob

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

def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)
MATPLOTLIB_FLAG = False
def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data
logger = logging
def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
  """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  name_key = (lambda _f: int(re.compile('model-(\d+)\.pt').match(_f).group(1)))
  time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
  sort_key = time_key if sort_by_time else name_key
  x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('_0.pth')], key=sort_key)
  to_del = [os.path.join(path_to_models, fn) for fn in
            (x_sorted('model')[:-n_ckpts_to_keep])]
  del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
  del_routine = lambda x: [os.remove(x), del_info(x)]
  rs = [del_routine(fn) for fn in to_del]



