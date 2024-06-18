from vqvae.model_vc import SynthesizerTrn
import os
import json
from vqvae.utils.data_utils import HParams
import torch
from gpt.model import UnifiedVoice

def load_model(model_name, model_path, config_path, device):
    config_path = os.path.expanduser(config_path)
    model_path = os.path.expanduser(model_path)
    if config_path.endswith('.json'):
        config = json.load(open(config_path))
    else:
        config = OmegaConf.load(config_path)
    if model_name=='vqvae':
        hps = HParams(**config)
        model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model)
        vqvae = torch.load(model_path, map_location=device)
        if 'model' in vqvae:
            sd = vqvae['model']
        if 'G' in vqvae:
            sd = vqvae['G']
        model.load_state_dict(sd, strict=True)
        model = model.to(device)
    elif model_name=='gpt':
        model = UnifiedVoice(**config['gpt'])
        gpt = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(gpt, strict=True)
        model = model.to(device)
    
    return model.eval()