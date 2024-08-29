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
        output_seq_len = latents.shape[2]*4 # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
        output_shape = (latents.shape[0], 128, output_seq_len)
        precomputed_embeddings = diffusion_model.timestep_independent(latents, conditioning_latents, output_seq_len, False)

        noise = torch.randn(output_shape, device=latents.device) * temperature
        mel = diffuser.p_sample_loop(diffusion_model, output_shape, noise=noise,
                                    model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                    progress=verbose)
        return mel[:,:,:output_seq_len]
class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)
MEL_MIN = -11.512925465
TORCH_MEL_MAX = 5.54517744448

def normalize_torch_mel(mel):
    return 2 * ((mel - MEL_MIN) / (TORCH_MEL_MAX - MEL_MIN)) - 1

def denormalize_torch_mel(norm_mel):
    return ((norm_mel+1)/2) * (TORCH_MEL_MAX - MEL_MIN) + MEL_MIN
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
        

        mel_channels = cfg['data']['n_mel_channels']
        self.mel_channels = mel_channels
        self.diffuser= SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [desired_diffusion_steps]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=False, conditioning_free_k=cond_free_k)
        self.infer_diffuser = SpacedDiffusion(use_timesteps=space_timesteps(trained_diffusion_steps, [50]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', trained_diffusion_steps),
                           conditioning_free=True, conditioning_free_k=cond_free_k, sampler='dpm++2m')
        self.diff_ref_enc = modules.MelStyleEncoder(
            mel_channels, style_vector_dim=gin_channels
        )
        self.diffusion = DiffusionTts(**cfg['diffusion'])
        self.in_proj = nn.Conv1d(mel_channels, inter_channels,3,1,1)
        self.enc_p = SpecEncoder(inter_channels, hidden_channels, filter_channels, True, n_heads,
                n_layers, kernel_size, p_dropout)
        self.enc_q = PosteriorEncoder(
            spec_channels, inter_channels, hidden_channels, True,
            5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.ref_enc = modules.MelStyleEncoder(
            mel_channels, style_vector_dim=gin_channels
        )
        self.quantizer = ResidualVectorQuantizer(dimension=inter_channels*4, n_q=1, bins=cfg['vaegan']['vq_bins'])
        self.gpt = UnifiedVoice(**cfg['gpt'])
        self.gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
        self.mel_loss_weight = cfg['train']['mel_weight']
        self.text_loss_weight = cfg['train']['text_weight']
        
        self.vq_enc = nn.Sequential(
                Transpose((1,2)),
                nn.LayerNorm(mel_channels),
                Transpose((1,2)),
                nn.Conv1d(mel_channels, inter_channels * 2, 3, 2, 1),
                nn.SiLU(),
                nn.Conv1d(inter_channels * 2, inter_channels * 4, 3,2,1),
                nn.SiLU(),
                nn.Conv1d(inter_channels * 4, inter_channels * 4, 3,1,1),
            )
        self.vq_dec = nn.Sequential(
                Transpose((1,2)),
                nn.LayerNorm(inter_channels * 4),
                Transpose((1,2)),
                nn.ConvTranspose1d(inter_channels * 4, inter_channels * 2,
                                   3, 2, padding=1, output_padding=1),
                nn.SiLU(),
                nn.ConvTranspose1d(inter_channels * 2, inter_channels,
                                   3, 2, padding=1, output_padding=1),
                nn.SiLU(),
                nn.Conv1d(inter_channels, mel_channels, 3,1,1)
            )
        self.vq_ref_enc = modules.MelStyleEncoder(
            mel_channels, style_vector_dim=inter_channels * 4
        )
        self.target = cfg['train']['target']
        if cfg['train']['target']=='vqvae':
            self.requires_grad_(False)
            self.vq_enc.requires_grad_(True)
            self.vq_dec.requires_grad_(True)
            self.vq_ref_enc.requires_grad_(True)
            self.quantizer.requires_grad_(True)
        if cfg['train']['target']=='gpt':
            self.requires_grad_(False)
            self.gpt.requires_grad_(True)
        if cfg['train']['target']=='diff':
            self.requires_grad_(False)
            self.diffusion.requires_grad_(True)
            self.diff_ref_enc.requires_grad_(True)
        if cfg['train']['target']=='flowvae':
            self.gpt.requires_grad_(False)
            self.diffusion.requires_grad_(False)
            self.diff_ref_enc.requires_grad_(False)
            self.vq_enc.requires_grad_(False)
            self.vq_dec.requires_grad_(False)
            self.vq_ref_enc.requires_grad_(False)
            self.quantizer.requires_grad_(False)
        # self.requires_grad_(False)
        # self.gpt.requires_grad_(True)
        # self.diffusion.requires_grad_(True)
    def forward_vq(self,y,y_lengths,data):
        assert y.shape[-1]%4==0
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        
        x_vq = self.vq_enc(y)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x_vq, layers=[0])
        g_vq = self.vq_ref_enc(y * y_mask, y_mask)
        quantized = quantized + g_vq
        recon = self.vq_dec(quantized)
        recon_loss = nn.L1Loss()(recon, y)
        vq_loss = commit_loss*0.25 + recon_loss
        return vq_loss
    def forward_diff(self,y,y_lengths,data):
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        # print(torch.max(y),torch.min(y))
        x_start = normalize_torch_mel(y)
        with torch.no_grad():
            code, _ = self.encode(data['raw_mel'], data['raw_spec_length'])
        t = torch.randint(0, self.desired_diffusion_steps, (x_start.shape[0],), device=y.device).long().to(y.device)
        with torch.no_grad():
            aligned_conditioning = self.gpt(
                data['raw_mel'], data['raw_spec_length'],
                data['text'], data['text_length'],
                code, data['raw_wav_length'],
                return_latent=True, clip_inputs=False).transpose(1,2)
        g_diff = self.diff_ref_enc(y * y_mask, y_mask)
        conditioning_latent = g_diff
        l_diff = self.diffuser.training_losses(
            model = self.diffusion, 
            x_start = x_start,
            t = t,
            model_kwargs = {
                "aligned_conditioning": aligned_conditioning,
                "conditioning_latent": conditioning_latent
            },
            )["loss"].mean()
        return l_diff
    def forward_gpt(self,y,y_lengths,data):
        with torch.no_grad():
            code, x_start = self.encode(data['raw_mel'], data['raw_spec_length'])
        input_params = [data['mel'], data['spec_length'],
            data['text'], data['text_length'], code, data['raw_wav_length']]
        loss_text, loss_mel, mel_logits = self.gpt(*input_params)
        loss_gpt = loss_text*self.text_loss_weight + loss_mel*self.mel_loss_weight
        return loss_gpt
        
    def forward_flowvae(self,y,y_lengths,data):
        assert y.shape[-1]%4==0
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        # print(y.shape)
        g = self.ref_enc(y * y_mask, y_mask)
        
        x = self.in_proj(y)
        x, m_p, logs_p = self.enc_p(x,y_lengths)
        quantized = x

        z, m_q, logs_q = self.enc_q(data['spec'], y_lengths,g)
        assert m_p.shape[-1]==m_q.shape[-1]
        z_p = self.flow(z, y_mask, g=g)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        l_diff=0
        loss_gpt=0
        vq_loss=0
        return (
            o,
            l_diff,
            loss_gpt,
            vq_loss,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )
    def forward(self, y, y_lengths, data):
        if self.target == 'vqvae':
            return self.forward_vq(y,y_lengths,data)
        elif self.target=='gpt':
            return self.forward_gpt(y,y_lengths,data)
        elif self.target=='diff':
            return self.forward_diff(y,y_lengths,data)
        elif self.target=='flowvae':
            return self.forward_flowvae(y,y_lengths,data)
        else:
            return self.forward_all(y,y_lengths,data)
    def forward_all(self, y, y_lengths, data):
        (
            o,
            l_diff,
            loss_gpt,
            vq_loss,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )  = self.forward_flowvae(y,y_lengths,data)
        l_diff = self.forward_diff(y,y_lengths,data)
        loss_gpt = self.forward_gpt(y,y_lengths,data)
        vq_loss = self.forward_vq(y,y_lengths,data)
        return (
            o,
            l_diff,
            loss_gpt,
            vq_loss,
            ids_slice,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, text, text_length, refer, refer_lengths, noise_scale=0.667):
        text = text[0].unsqueeze(0)
        text_length = text_length[0].unsqueeze(0)
        refer = refer[0].unsqueeze(0)
        refer_lengths = refer_lengths[0].unsqueeze(0)
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
        # print(codes.shape,codes)
        codes = codes[:,:-1]
        latent = self.gpt(refer, refer_lengths, text,
            text_length, codes,
            torch.tensor([codes.shape[-1]*self.gpt.mel_length_compression], device=text.device),
            return_latent=True, clip_inputs=False).transpose(1,2)
        g_diff = self.diff_ref_enc(refer * refer_mask, refer_mask)
        latent = do_spectrogram_diffusion(self.diffusion, self.infer_diffuser,latent,g_diff,temperature=1.0)
        latent = denormalize_torch_mel(latent)
        y_lengths = torch.LongTensor([latent.shape[-1]]).to(latent.device)
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, latent.size(2)), 1).to(latent.dtype)
        o = self.infer_flowvae(latent,y_lengths,None)
        return  o
    def infer_gpt(self, text, text_length, refer, refer_lengths, noise_scale=0.667):
        text = text[0].unsqueeze(0)
        text_length = text_length[0].unsqueeze(0)
        refer = refer[0].unsqueeze(0)
        refer_lengths = refer_lengths[0].unsqueeze(0)
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
        # print(codes)
        latent = self.quantizer.decode(codes[:,:-1].unsqueeze(0))
        # print(latent.shape)
        
        # print(latent)
        if latent.shape[-1]==0:
            latent = torch.zeros(latent.shape[0],latent.shape[1],16).to(latent.device)
        
        y_lengths = torch.LongTensor([latent.shape[-1]*4]).to(latent.device)
        # print(codes,latent)
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, latent.size(2)*4), 1).to(latent.dtype)
        g_vq = self.vq_ref_enc(refer * refer_mask, refer_mask)
        latent = latent + g_vq
        recon = self.vq_dec(latent)
        # print(recon.shape, y_lengths)
        o = self.infer_flowvae(recon,y_lengths,None)
        return  o
    def infer_flowvae(self, y, y_lengths,data,noise_scale=0.667):
        y = y[0].unsqueeze(0)
        y_lengths = y_lengths[0].unsqueeze(0)
        assert y.shape[-1]%4==0
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        quantized_mask = torch.unsqueeze(commons.sequence_mask(y_lengths//4, y.size(2)//4), 1).to(y.dtype)
        g = self.ref_enc(y * y_mask, y_mask)
        x = self.in_proj(y)
        x, m_p, logs_p = self.enc_p(x,y_lengths)
        
        # x, m_p, logs_p = self.enc_p[4](x,y_lengths)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec(z, g=g)
        return o
    def infer_vqvae(self,y):
        y = y[0].unsqueeze(0)
        assert y.shape[-1]%4==0
        x = self.vq_enc(y)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x, layers=[0])
        y_lengths = torch.LongTensor([int(y.shape[-1])]).to(y.device)
        y_mask = torch.unsqueeze( commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        g_vq = self.vq_ref_enc(y * y_mask, y_mask)
        quantized = quantized + g_vq
        recon = self.vq_dec(quantized)
        y_lengths = torch.LongTensor([y.shape[-1]]).to(y.device)
        o = self.infer_flowvae(recon,y_lengths,None)
        return  recon,o
    def encode(self, y, y_lengths):
        x_vq = self.vq_enc(y)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x_vq, layers=[0])
        return codes[0].detach(), x_vq.detach()
   

