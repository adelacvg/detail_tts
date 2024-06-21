import logging
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.WARNING)
import random
import copy
import time
import torch.autograd.profiler as profiler
import math
import torch
from torch import nn
from torch.nn import functional as F
import vqvae.modules.commons as commons
from vqvae.modules import modules, attentions
from vqvae.utils.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule
from vqvae.diff_model import DiffusionTts
from gpt.model import UnifiedVoice

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from vqvae.modules.commons import init_weights, get_padding
from vqvae.modules.quantize import ResidualVectorQuantizer


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.text_embedding = nn.Embedding(256, hidden_channels)
        nn.init.normal_(self.text_embedding.weight, 0.0, hidden_channels**-0.5)
        # self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, text=None, text_lengths=None):
        text_mask = torch.unsqueeze(
            commons.sequence_mask(text_lengths, text.size(1)), 1
        ).to(text.dtype)
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder(text * text_mask, text_mask)
        return text, text_mask
        # stats = self.proj(text) * text_mask
        # m, logs = torch.split(stats, self.out_channels, dim=1)
        # return text, m, logs

class SpecEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        sample,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        gin_channels = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.sample = sample
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.gin_channels = gin_channels

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.out_proj = nn.Conv1d(hidden_channels, out_channels, 1)
        if self.gin_channels is not None:
            self.ge_proj = nn.Linear(gin_channels,hidden_channels)
        if self.sample==True:
            self.proj = nn.Conv1d(out_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, g=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if g is not None:
            y = y + self.ge_proj(g.squeeze(-1)).unsqueeze(-1)
        y = self.encoder(y * y_mask, y_mask)
        y = self.out_proj(y)
        if self.sample==False:
            return y

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        sample,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.sample = sample

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        if self.sample==True:
            self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        if self.sample == False:
            return x
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            l.remove_weight_norm()
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def do_spectrogram_diffusion(diffusion_model, diffuser, latents, conditioning_latents, temperature=1, verbose=True):
    """
    Uses the specified diffusion model to convert discrete codes into a spectrogram.
    """
    with torch.no_grad():
        output_seq_len = latents.shape[2] # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 192, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                    model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                    progress=verbose)
        return mel[:,:,:output_seq_len]
class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        prosody_size=20,
        n_speakers=0,
        gin_channels=0,
        semantic_frame_rate=None,
        cfg=None,
        **kwargs
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.mel_size = prosody_size
        trained_diffusion_steps = 4000
        self.trained_diffusion_steps = 4000
        desired_diffusion_steps = 200
        self.desired_diffusion_steps = 200
        cond_free_k = 2.

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        
        self.enc_p = []
        self.enc_p.extend(
            [
                PosteriorEncoder(
                    spec_channels, inter_channels, hidden_channels, False,
                5, 1, 16, gin_channels=gin_channels),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, False, n_heads,
                n_layers, kernel_size, p_dropout,gin_channels=gin_channels),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, False, n_heads,
                n_layers, kernel_size, p_dropout,gin_channels=gin_channels),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, False, n_heads,
                n_layers, kernel_size, p_dropout),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, True, n_heads,
                n_layers, kernel_size, p_dropout),
            ]
        )
        self.diffuser= SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=False, conditioning_free_k=cond_free_k)
        self.infer_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=cond_free_k, sampler='dpm++2m')
        self.diffusion = DiffusionTts(**cfg['diffusion'])
        self.enc_p = nn.ModuleList(self.enc_p)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, True,
            5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )
        self.proj = []
        self.proj.extend([nn.Conv1d(inter_channels, inter_channels, kernel_size=2, stride=2) for _ in range(2)])
        self.proj = nn.ModuleList(self.proj)
        self.proj_out = []
        self.proj_out.extend([nn.ConvTranspose1d(inter_channels, inter_channels, kernel_size=2, stride=2) for _ in range(2)])
        self.proj_out = nn.ModuleList(self.proj_out)
        self.quantizer = ResidualVectorQuantizer(dimension=hidden_channels, n_q=1, bins=1024)
        self.gpt = UnifiedVoice(**cfg['gpt'])
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
        self.mel_loss_weight = cfg['train']['mel_weight']
        self.text_loss_weight = cfg['train']['text_weight']
        # self.code_emb.requires_grad_(False)
        # self.quantizer.requires_grad_(False)
        # self.diffusion.requires_grad_(False)

    def forward(self, y, y_lengths, data):
        assert y.shape[-1]%4==0
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        quantized_mask = torch.unsqueeze(commons.sequence_mask(y_lengths//4, y.size(2)//4), 1).to(y.dtype)
        g = self.ref_enc(y * y_mask, y_mask)

        x = self.enc_p[0](y, y_lengths, g=g)
        x = self.proj[0](x)
        x = self.enc_p[1](x, y_lengths//2, g=g)
        x = self.proj[1](x)
        x = self.enc_p[2](x,y_lengths//4, g=g)
        l_diff=0
        commit_loss=0
        quantized, codes, commit_loss, quantized_list = self.quantizer(x.detach(), layers=[0])

        with torch.no_grad():
            code, x_start = self.encode(data['raw_spec'], data['raw_spec_length'])
        input_params = [data['raw_spec'], data['raw_spec_length'],
            data['text'], data['text_length'], code, data['raw_wav_length']]
        loss_text, loss_mel, mel_logits = self.gpt(*input_params)
        loss_gpt = loss_text*self.text_loss_weight + loss_mel*self.mel_loss_weight

        t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=x.device).long().to(x.device)
        with torch.no_grad():
            aligned_conditioning = self.gpt(
                data['raw_spec'], data['raw_spec_length'],
                data['text'], data['text_length'],
                code, data['raw_wav_length'],
                return_latent=True, clip_inputs=False).transpose(1,2)
        conditioning_latent = g.detach()
        l_diff = self.diffuser.training_losses( 
            model = self.diffusion, 
            x_start = x_start,
            t = t,
            model_kwargs = {
                "aligned_conditioning": aligned_conditioning,
                "conditioning_latent": conditioning_latent
            },
            )["loss"].mean()

        x = self.proj_out[0](x)
        x = self.enc_p[3](x,y_lengths//2)
        x = self.proj_out[1](x)
        x, m_p, logs_p = self.enc_p[4](x,y_lengths)
        quantized = x

        z, m_q, logs_q = self.enc_q(y, y_lengths,g)
        assert m_p.shape[-1]==m_q.shape[-1]
        z_p = self.flow(z, y_mask, g=g)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_diff,
            loss_gpt,
            commit_loss,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, text, text_length, refer, refer_lengths, noise_scale=0.667):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        g = self.ref_enc(refer * refer_mask, refer_mask)
        codes = self.gpt.inference_speech_tortoise(
                    refer,
                    refer_lengths,
                    text,
                    do_sample=True,
                    top_p=0.8,
                    temperature=0.8,
                    num_return_sequences=1,
                    length_penalty=1.0,
                    repetition_penalty=2.0,
                    max_generate_length=600)
        latent = self.gpt(refer, refer_lengths, text,
            text_length, codes,
            torch.tensor([codes.shape[-1]*self.gpt.mel_length_compression], device=text.device),
            return_latent=True, clip_inputs=False).transpose(1,2)

        latent = do_spectrogram_diffusion(self.diffusion, self.infer_diffuser,latent,g,temperature=1.0)
        y_lengths = torch.LongTensor([latent.shape[-1]*4]).to(latent.device)
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, latent.size(2)*4), 1).to(latent.dtype)
        x = self.proj_out[0](latent)
        x = self.enc_p[3](x,y_lengths//2)
        x = self.proj_out[1](x)
        x, m_p, logs_p = self.enc_p[4](x,y_lengths)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z, g=g)
        return  o
    def encode(self, y, y_lengths):
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        g = self.ref_enc(y * y_mask, y_mask)
        x = self.enc_p[0](y, y_lengths, g=g)
        x = self.proj[0](x)
        x = self.enc_p[1](x, y_lengths//2, g=g)
        x = self.proj[1](x)
        x = self.enc_p[2](x,y_lengths//4, g=g)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x.detach(), layers=[0])
        return codes[0].detach(), x.detach()
   
        

