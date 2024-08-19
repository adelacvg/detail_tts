import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
import copy
import random
import time
from datetime import datetime
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vqvae.utils.log_utils import clean_checkpoints, plot_spectrogram_to_numpy, summarize
from typing import List, Optional, Tuple, Union
import torch
import os
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from accelerate import Accelerator
from vqvae.model_24k import SynthesizerTrn, MultiPeriodDiscriminator
from vqvae.dataset_24k import TextAudioCollate, TextAudioSpeakerLoader
from vqvae.utils.data_utils import spec_to_mel_torch, mel_spectrogram_torch, HParams, spectrogram_torch
from vqvae.utils import utils
import vqvae.modules.commons as commons
import torchaudio
from vqvae.modules.losses import generator_loss, discriminator_loss, feature_loss, kl_loss
from torchaudio.functional import phase_vocoder, resample, spectrogram
from torchaudio import transforms
from datetime import timedelta
from accelerate import InitProcessGroupKwargs

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            else:
                # print(name)
                continue
        except Exception as e:
            print(e)
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def cycle(dl):
    while True:
        for data in dl:
            yield data
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
from accelerate import DistributedDataParallelKwargs
class Trainer(object):
    def __init__(self, cfg_path):

        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        # Create the custom configuration
        process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))  # 1.5 hours
        self.accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
        self.cfg = json.load(open(cfg_path))
        hps = HParams(**self.cfg)
        self.hps = hps
        self.config = hps
        dataset = TextAudioSpeakerLoader(hps)
        collate_fn = TextAudioCollate()
        self.dataloader = DataLoader(
            dataset,
            batch_size=hps.train.batch_size,
            num_workers=hps.train.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            prefetch_factor=16,)
        self.train_steps = self.cfg['train']['train_steps']
        self.val_freq = self.cfg['train']['val_freq']
        if self.accelerator.is_main_process:
            now = datetime.now()
            self.logs_folder = Path(self.cfg['train']['logs_folder']+'/'+now.strftime("%Y-%m-%d-%H-%M-%S"))
            self.logs_folder.mkdir(exist_ok = True, parents=True)
        self.G = SynthesizerTrn(hps.data.filter_length // 2 + 1,hps.train.segment_size // hps.data.hop_length, **hps.vaegan, cfg=hps)
        self.D = MultiPeriodDiscriminator()

        # self.G.requires_grad_(False)
        # self.G.gpt.requires_grad_(False)
        # self.G.gpt.requires_grad_(True)
        # self.G.diffusion.requires_grad_(True)
        # self.G.dec.requires_grad_(True)
        if self.hps.train.target in ['gpt','diff','vqvae']:
            self.D.requires_grad_(False)

        print("G params:", count_parameters(self.G))
        print("D params:", count_parameters(self.D))
        self.G_optimizer = AdamW(self.G.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        self.D_optimizer = AdamW(self.D.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.G_optimizer, gamma=hps.train.lr_decay, last_epoch=-1
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.D_optimizer, gamma=hps.train.lr_decay, last_epoch=-1
        )
        self.G, self.G_optimizer, self.D, self.D_optimizer, self.dataloader, self.scheduler_g, self.scheduler_d = self.accelerator.prepare(
            self.G, self.G_optimizer, self.D, self.D_optimizer, self.dataloader, self.scheduler_g, self.scheduler_d)
        self.dataloader = cycle(self.dataloader)
        self.step=0
        self.epoch=1
        self.gradient_accumulate_every=self.hps.train.gradient_accumulate_every
        # self.aug = Augment(hps)
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'epoch': self.epoch,
            'G': self.accelerator.get_state_dict(self.G),
            'D': self.accelerator.get_state_dict(self.D),
            'G_opt': self.accelerator.get_state_dict(self.G_optimizer),
            'D_opt': self.accelerator.get_state_dict(self.D_optimizer)
        }
        torch.save(data, str(self.logs_folder / f'model-{milestone}.pt'))
    def load(self, model_path):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(model_path, map_location=device)
        G_state_dict = data['G']
        D_state_dict = data['D']
        G_opt_state_dict = data['G_opt']
        D_opt_state_dict = data['D_opt']
        self.step = data['step']
        self.epoch = data['epoch']
        G = accelerator.unwrap_model(self.G)
        current_model_dict = G.state_dict()
        G_state_dict={k:v if v.size()==current_model_dict[k].size()
            # and 'diff' not in k
            # and 'gpt' not in k
            # and 'vq' not in k
            else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), G_state_dict.values())}
        G.load_state_dict(G_state_dict, strict=False)
        D = accelerator.unwrap_model(self.D)
        D.load_state_dict(D_state_dict)
        G_opt = accelerator.unwrap_model(self.G_optimizer)
        try:
            G_opt.load_state_dict(G_opt_state_dict)
        except:
            # G.quantizer.layers[0]._codebook.inited=torch.Tensor([True])
            print('Fail to load G_opt')
        D_opt = accelerator.unwrap_model(self.D_optimizer)
        D_opt.load_state_dict(D_opt_state_dict)
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        hps = self.hps
        cnt=0

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=self.logs_folder)
        epoch=self.epoch
        for _ in range(self.epoch):
            self.scheduler_g.step()
            self.scheduler_d.step()
        # cnt=0
        # for data in self.dataloader:
        #     cnt+=1
        #     print(cnt)
        #     if cnt>45000:
        #         break
        with tqdm(initial = self.step, total = self.train_steps, disable = not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:
                # self.dataloader.batch_sampler.epoch=epoch
                # for data in self.dataloader:
                
                # data = next(self.dataloader)
                # for d in data:
                #     data[d] = data[d].to(device)
                # spec, y, spec_length, wav = data['spec'],data['wav'],data['spec_length'],data['raw_wav']
                
                # with torch.autograd.detect_anomaly():
                if self.hps.train.target in ['vqvae','gpt','diff']:
                    total_loss = 0.
                    for _ in range(self.gradient_accumulate_every):
                        data = next(self.dataloader)
                        for d in data:
                            data[d] = data[d].to(device)
                        spec, y, spec_length, wav = data['spec'],data['wav'],data['spec_length'],data['raw_wav']
                        with self.accelerator.autocast():
                            loss = self.G(data['mel'], spec_length,data=data)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()
                        self.accelerator.backward(loss)
                    grad_norm = get_grad_norm(self.G)
                    accelerator.clip_grad_norm_(self.G.parameters(), 1.0)
                    pbar.set_description(f'{self.hps.train.target}_loss: {total_loss:.4f}')
                    accelerator.wait_for_everyone()
                    self.G_optimizer.step()
                    self.G_optimizer.zero_grad()
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process and self.step % self.val_freq == 0 and self.hps.train.target in ['diff','gpt']:
                        eval_model = self.accelerator.unwrap_model(self.G)
                        eval_model.eval()
                        with torch.no_grad():
                            if self.hps.train.target == 'diff':
                                wav_eval = eval_model.infer(data['text'], data['text_length'], data['mel'], spec_length)
                            if self.hps.train.target == 'gpt':
                                wav_eval = eval_model.infer_gpt(data['text'], data['text_length'], data['mel'], spec_length)
                        eval_model.train()
                        milestone = self.step // self.cfg['train']['save_freq'] 
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), hps.data.sampling_rate)
                        torchaudio.save(str(self.logs_folder / f'gt-{milestone}.wav'), wav[0].detach().cpu(), hps.data.sampling_rate)
                    if accelerator.is_main_process and self.step % self.val_freq == 0 and self.hps.train.target == 'vqvae':
                        eval_model = self.accelerator.unwrap_model(self.G)
                        eval_model.eval()
                        with torch.no_grad():
                            mel_recon,wav_recon = eval_model.infer_vqvae(data['raw_mel'])
                        eval_model.train()
                        image_dict = {
                            "img/mel_raw": plot_spectrogram_to_numpy(data['raw_mel'][0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                            "img/mel_pred": plot_spectrogram_to_numpy(mel_recon[0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                        }
                        summarize(
                            writer=writer,
                            global_step=self.step,
                            images=image_dict
                        )
                        milestone = self.step // self.cfg['train']['save_freq'] 
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_recon[0].detach().cpu(), hps.data.sampling_rate)
                        torchaudio.save(str(self.logs_folder / f'gt-{milestone}.wav'), wav[0].detach().cpu(), hps.data.sampling_rate)
                elif self.hps.train.target=='flowvae':
                    data = next(self.dataloader)
                    for d in data:
                        data[d] = data[d].to(device)
                    spec, y, spec_length, wav = data['spec'],data['wav'],data['spec_length'],data['raw_wav']
                    with self.accelerator.autocast():
                        y_hat, diff_loss, loss_gpt, commit_loss, ids_slice, z_mask,\
                        (z, z_p, m_p, logs_p, m_q, logs_q),\
                        latent = self.G(data['mel'], spec_length,data=data)
                        mel = spec_to_mel_torch(
                            spec,
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                        y_mel = commons.slice_segments(
                            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                        )
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )

                        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
                        # Discriminator
                        y_d_hat_r, y_d_hat_g, _, _ = self.D(y, y_hat.detach())
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
                    self.D_optimizer.zero_grad()
                    self.accelerator.backward(loss_disc_all)
                    grad_norm_d = commons.clip_grad_value_(self.D.parameters(), None)
                    accelerator.wait_for_everyone()
                    self.D_optimizer.step()
                    accelerator.wait_for_everyone()

                    # Generator
                    with self.accelerator.autocast():
                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.D(y, y_hat)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel \
                        + loss_kl + commit_loss + diff_loss + loss_gpt
                    # print(loss_gen, loss_fm, loss_mel, loss_kl, commit_loss, diff_loss, loss_gpt)
                    self.G_optimizer.zero_grad()
                    self.accelerator.backward(loss_gen_all)

                    grad_norm_g = commons.clip_grad_value_(self.G.parameters(), None)
                    get_grad_norm(self.G)
                    accelerator.wait_for_everyone()
                    self.G_optimizer.step()
                    accelerator.wait_for_everyone()
                    pbar.set_description(f'mel_loss:{loss_mel:.4f} kl_loss:{loss_kl:.4f}')
                    #infer flowvae
                    if accelerator.is_main_process and self.step % self.val_freq == 0:
                        eval_model = self.accelerator.unwrap_model(self.G)
                        eval_model.eval()
                        with torch.no_grad():
                            wav_eval = eval_model.infer_flowvae(data['raw_mel'], data['raw_spec_length'],data=data)
                        eval_model.train()
                        milestone = self.step // self.cfg['train']['save_freq']
                        # print(data['raw_spec_length'],data['raw_wav_length'],wav_eval[0].shape[-1])
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), hps.data.sampling_rate)
                        # torchaudio.save(str(self.logs_folder / f'raw-{milestone}.wav'), data['raw_wav'][0].detach().cpu(), hps.data.sampling_rate)
                else:
                    data = next(self.dataloader)
                    for d in data:
                        data[d] = data[d].to(device)
                    spec, y, spec_length, wav = data['spec'],data['wav'],data['spec_length'],data['raw_wav']
                    with self.accelerator.autocast():
                        y_hat, diff_loss, loss_gpt, commit_loss, ids_slice, z_mask,\
                        (z, z_p, m_p, logs_p, m_q, logs_q),\
                        latent = self.G(data['mel'], spec_length,data=data)
                        mel = spec_to_mel_torch(
                            spec,
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )
                        y_mel = commons.slice_segments(
                            mel, ids_slice, hps.train.segment_size // hps.data.hop_length
                        )
                        y_hat_mel = mel_spectrogram_torch(
                            y_hat.squeeze(1),
                            hps.data.filter_length,
                            hps.data.n_mel_channels,
                            hps.data.sampling_rate,
                            hps.data.hop_length,
                            hps.data.win_length,
                            hps.data.mel_fmin,
                            hps.data.mel_fmax,
                        )

                        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)  # slice
                        # Discriminator
                        y_d_hat_r, y_d_hat_g, _, _ = self.D(y, y_hat.detach())
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
                    loss_disc_all = loss_disc
                    self.D_optimizer.zero_grad()
                    self.accelerator.backward(loss_disc_all)
                    grad_norm_d = commons.clip_grad_value_(self.D.parameters(), None)
                    accelerator.wait_for_everyone()
                    self.D_optimizer.step()
                    accelerator.wait_for_everyone()

                    # unused_params =[]
                    # G_ = self.accelerator.unwrap_model(self.G)
                    # unused_params.extend(list(G_.quantized_detail_enc.parameters()))
                    # extraneous_addition = 0
                    # for p in unused_params:
                    #     extraneous_addition = extraneous_addition + p.mean()
                    # Generator
                    with self.accelerator.autocast():
                        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.D(y, y_hat)
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel \
                        + loss_kl + commit_loss + diff_loss + loss_gpt
                    # print(loss_gen, loss_fm, loss_mel, loss_kl, commit_loss, diff_loss, loss_gpt)
                    self.G_optimizer.zero_grad()
                    self.accelerator.backward(loss_gen_all)

                    grad_norm_g = commons.clip_grad_value_(self.G.parameters(), None)
                    get_grad_norm(self.G)
                    accelerator.wait_for_everyone()
                    self.G_optimizer.step()
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        print(f"vq_loss:{commit_loss:.4f} mel_loss:{loss_mel:.4f} kl_loss:{loss_kl:.4f} diff_loss:{diff_loss:.4f} gpt_loss:{loss_gpt:.4f}")
                    pbar.set_description(f'G_loss:{loss_gen_all:.4f} D_loss:{loss_disc_all:.4f}')
                    if accelerator.is_main_process and self.step % self.val_freq == 0:
                        lr = self.G_optimizer.param_groups[0]["lr"]
                        eval_model = self.accelerator.unwrap_model(self.G)
                        eval_model.eval()
                        with torch.no_grad():
                            wav_eval = eval_model.infer(data['text'], data['text_length'], data['mel'], spec_length)
                        eval_model.train()
                        milestone = self.step // self.cfg['train']['save_freq'] 
                        torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), hps.data.sampling_rate)
                        scalar_dict = {
                                "gen/loss_gen_all": loss_gen_all,
                                "gen/loss_gen":loss_gen,
                                'gen/loss_fm':loss_fm,
                                'gen/loss_mel':loss_mel,
                                'gen/loss_kl':loss_kl, 
                                'gen/loss_diff':diff_loss, 
                                'gen/loss_gpt':loss_gpt, 
                                'gen/loss_commit':commit_loss,
                                "norm/G_grad": grad_norm_g, 
                                "norm/D_grad": grad_norm_d,
                                'disc/loss_disc_all':loss_disc_all,
                                'gen/lr':lr,
                            }
                        image_dict = {
                            "img/mel": plot_spectrogram_to_numpy(y_mel[0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                            "img/mel_pred": plot_spectrogram_to_numpy(y_hat_mel[0, :, :].detach().unsqueeze(-1).cpu().numpy()),
                            "img/mel_raw": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                            "img/latent": plot_spectrogram_to_numpy(latent[0].data.cpu().numpy()),
                        }
                        audios_dict = {
                            'wav/gt':wav[0,0,:].detach().cpu(),
                            'wav/pred':wav_eval[0].detach().cpu()
                        }
                        summarize(
                            writer=writer,
                            global_step=self.step,
                            images=image_dict,
                            audios=audios_dict,
                            scalars=scalar_dict,
                            audio_sampling_rate=hps.data.sampling_rate
                        )
                if accelerator.is_main_process and self.step % self.cfg['train']['save_freq']==0:
                    keep_ckpts = self.cfg['train']['keep_ckpts']
                    if keep_ckpts > 0:
                        clean_checkpoints(path_to_models=self.logs_folder, n_ckpts_to_keep=keep_ckpts, sort_by_time=True)
                    self.save(self.step//1000)
                self.step += 1
                pbar.update(1)
                if self.step%50000==0:
                    self.scheduler_g.step()
                    self.scheduler_d.step()
                    epoch = epoch + 1
        accelerator.print('training complete')


if __name__ == '__main__':
    trainer = Trainer(cfg_path='vqvae/configs/config_24k.json')
    trainer.load('/home/hyc/detail_tts/logs/2024-08-19-14-46-30/model-474.pt')
    trainer.train()
