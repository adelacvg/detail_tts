import torch
import random
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

import vqvae.modules.attentions as attentions
import vqvae.modules.commons as commons
import vqvae.modules.modules as modules
from vqvae.utils import utils
from vqvae.modules.commons import get_padding
from vqvae.modules.quantize import ResidualVectorQuantizer
from vqvae.utils.utils import f0_to_coarse


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

class Encoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        # print(x.shape,x_lengths.shape)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

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

    def forward(self, y, y_lengths, g=None, refer=None, refer_lengths=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(
            y.dtype
        )
        if g is not None:
            y = y + self.ge_proj(g.squeeze(-1)).unsqueeze(-1)
        if refer is not None:
            y_mask2 = torch.unsqueeze(commons.sequence_mask(y_lengths+refer_lengths, y.size(2)+refer.shape[2]), 1).to(
                y.dtype
            )
            T = y.shape[-1]
            y = torch.cat([y,refer],dim=2)
            y = self.encoder(y * y_mask2, y_mask2)
            y = y[:,:,:T]
        else:
            y = self.encoder(y * y_mask, y_mask)
        y = self.out_proj(y)
        if self.sample==False:
            return y*y_mask

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs

class TextEncoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 n_layers,
                 gin_channels=0,
                 filter_channels=None,
                 n_heads=None,
                 p_dropout=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
        self.f0_emb = nn.Embedding(256, hidden_channels)

        self.enc_ = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

    def forward(self, x, x_mask, f0=None, noice_scale=1):
        x = x + self.f0_emb(f0).transpose(1, 2)
        x = self.enc_(x * x_mask, x_mask)
        stats = x
        x = self.proj(x) * x_mask
        m, logs = torch.split(x, self.out_channels, dim=1)

        return stats, m, logs, x_mask


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
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
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
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
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
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


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed

class F0Decoder(nn.Module):
    def __init__(self,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 spk_channels=0):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = attentions.FFT(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += norm_f0
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self,
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
                 gin_channels,
                 ssl_dim,
                 sampling_rate=44100,
                 vol_embedding=False,
                 vocoder_name = "nsf-hifigan",
                 use_depthwise_conv = False,
                 use_automatic_f0_prediction = True,
                 flow_share_parameter = False,
                 n_flow_layer = 4,
                 n_layers_trans_flow = 3,
                 use_transformer_flow = False,
                 **kwargs):

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
        self.gin_channels = gin_channels
        self.ssl_dim = ssl_dim
        self.vol_embedding = vol_embedding
        self.use_depthwise_conv = use_depthwise_conv
        self.use_automatic_f0_prediction = use_automatic_f0_prediction
        self.n_layers_trans_flow = n_layers_trans_flow
        if vol_embedding:
           self.emb_vol = nn.Linear(1, hidden_channels)

        self.pre = nn.Conv1d(spec_channels, hidden_channels, kernel_size=5, padding=2)

        self.enc_p = []
        self.enc_p.extend(
            [
                SpecEncoder( inter_channels, hidden_channels, filter_channels, False, n_heads,
                4, kernel_size, p_dropout,gin_channels=gin_channels),
                SpecEncoder(
                    inter_channels, hidden_channels, filter_channels, True, n_heads,
                6, kernel_size, p_dropout,gin_channels=gin_channels),
            ]
        )
        self.enc_p = nn.ModuleList(self.enc_p)
        self.spec_proj = nn.Conv1d(spec_channels, hidden_channels, 1)

        self.enc_text = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels=filter_channels,
            n_heads=n_heads,
            n_layers=n_layers,
            kernel_size=kernel_size,
            p_dropout=p_dropout
        )
        hps = {
            "sampling_rate": sampling_rate,
            "inter_channels": inter_channels,
            "resblock": resblock,
            "resblock_kernel_sizes": resblock_kernel_sizes,
            "resblock_dilation_sizes": resblock_dilation_sizes,
            "upsample_rates": upsample_rates,
            "upsample_initial_channel": upsample_initial_channel,
            "upsample_kernel_sizes": upsample_kernel_sizes,
            "gin_channels": gin_channels,
            "use_depthwise_conv":use_depthwise_conv
        }
        
        from vqvae.vdecoder.hifigan.models import Generator
        self.dec = Generator(h=hps)

        self.enc_q = Encoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)

        self.ref_enc = modules.MelStyleEncoder(
            spec_channels, style_vector_dim=gin_channels
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        self.f0_decoder = F0Decoder(
            1,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            spk_channels=gin_channels
        )
        self.f0_detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            4,
            kernel_size,
            p_dropout,
            gin_channels = gin_channels,
        )
        self.detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            6,
            kernel_size,
            p_dropout,
            gin_channels = gin_channels,
        )
        self.quantized_detail_enc = SpecEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            False,
            n_heads,
            6,
            kernel_size,
            p_dropout,
            gin_channels = gin_channels,
        )
        self.ssl_proj = nn.ModuleList([
            modules.WN(hidden_channels, 5, 1, 16, gin_channels=gin_channels),
            nn.Conv1d(hidden_channels, hidden_channels, 3, 2, 1),
            SpecEncoder(inter_channels, hidden_channels, filter_channels, False, n_heads, 3, kernel_size, p_dropout,gin_channels = gin_channels),
            nn.Conv1d(inter_channels, inter_channels, 3, 2, 1),
            SpecEncoder(inter_channels, hidden_channels, filter_channels, False, n_heads, 3, kernel_size, p_dropout,gin_channels = gin_channels),
            nn.ConvTranspose1d(inter_channels, inter_channels, 3, 2, 1,output_padding=1),
            SpecEncoder(inter_channels, hidden_channels, filter_channels, False, n_heads, 3, kernel_size, p_dropout,gin_channels = gin_channels),
            nn.ConvTranspose1d(inter_channels, inter_channels, 3, 2, 1,output_padding=1),
            SpecEncoder(inter_channels, hidden_channels, filter_channels, True, n_heads, 3, kernel_size, p_dropout,gin_channels = gin_channels),]
            )
        self.quantizer = ResidualVectorQuantizer(dimension=hidden_channels, n_q=8, bins=1024)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.code_emb = nn.Embedding(1024, hidden_channels)
        nn.init.normal_(self.code_emb.weight, 0.0, hidden_channels**-0.5)

        # self.quantizer.requires_grad_(False)

    def forward(self, spec, spec_lengths, f0, uv):
        quantized_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths//4, spec.size(2)//4), 1).to(spec.dtype)
        spec_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype)
        half_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths//2, spec.size(2)//2), 1).to(spec.dtype)
        x_mask = spec_mask
        g = self.ref_enc(spec,spec_mask)

        commit_loss=0
        # ssl prenet
        x = self.pre(spec) * x_mask
        x1 = self.ssl_proj[0](x,spec_mask,g)
        x2 = self.ssl_proj[1](x1)
        x3 = self.ssl_proj[2](x2,spec_lengths//2,g)
        x4 = self.ssl_proj[3](x3)
        x5 = self.ssl_proj[4](x4,spec_lengths//4,g)
        l_quantized_detail = 0
        quantized, codes, commit_loss, quantized_list = self.quantizer(x5, layers=[0,1,2,3,4,5,6,7])

        base = self.code_emb(codes[0]).transpose(1,2)
        noise = ((x5.detach()-base)//100).detach()
        # f = random.randint(0,2)
        # if f==0:
        #     i = 0
        # else:
        i = random.randint(0,100)
        quantized_detail = base+noise*i+torch.randn_like(base)*0.005
        quantized_detail_ = self.quantized_detail_enc(quantized_detail, spec_lengths//4, g=g)
        l_quantized_detail += torch.sum(((noise - quantized_detail_)**2)*quantized_mask) / torch.sum(quantized_mask)
        x6 = self.ssl_proj[5](x5)
        x7 = self.ssl_proj[6](x6,spec_lengths//2,g)
        x8 = self.ssl_proj[7](x7)
        x9 = self.ssl_proj[8](x8,spec_lengths,g)
        x = x9

        l_residual_detail = 0
        
        # f0 predict
        lf0 = 2595. * torch.log10(1. + f0.unsqueeze(1) / 700.) / 500
        norm_lf0 = utils.normalize_f0(lf0, x_mask, uv)
        f0_detail = self.f0_prenet(norm_lf0)
        f0_detail_ = self.f0_detail_enc(x.detach(), spec_lengths, g=g)
        l_f0_detail = torch.sum(((f0_detail - f0_detail_) ** 2)*x_mask) / torch.sum(x_mask)
        pred_lf0 = self.f0_decoder(x, f0_detail, x_mask, spk_emb=g)

        # encoder
        x, m_t, logs_t, _ = self.enc_text(x, x_mask, f0=f0_to_coarse(f0))
        spec_mask = x_mask
        spec2 = self.spec_proj(spec)*spec_mask
        detail = self.enc_p[0](spec2, spec_lengths, g=g)
        detail_ = self.detail_enc(x, spec_lengths, g=g)
        l_detail = torch.sum(((detail - detail_) ** 2)*spec_mask) / torch.sum(spec_mask)
        x = x + detail
        x, m_p, logs_p = self.enc_p[1](x, spec_lengths, g=g)
        latent = x

        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        # flow
        z_p = self.flow(z, spec_mask, g=g)
        z_slice, pitch_slice, ids_slice = commons.rand_slice_segments_with_pitch(z, f0, spec_lengths, self.segment_size)

        # nsf decoder
        o = self.dec(z_slice, g=g, f0=pitch_slice)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q, m_t, logs_t), pred_lf0, lf0, l_detail, l_f0_detail, latent, commit_loss, l_quantized_detail, l_residual_detail

    @torch.no_grad()
    def infer(self, c, c_lengths, refer, refer_lengths, f0=None, noice_scale=0.35, predict_f0=True):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        quantized_mask = torch.unsqueeze(commons.sequence_mask(c_lengths//4, c.size(2)//4), 1).to(c.dtype)
        c_mask = torch.unsqueeze(
            commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        g = self.ref_enc(refer, refer_mask)
        
        x_mask = torch.unsqueeze(commons.sequence_mask(c_lengths, c.size(2)), 1).to(c.dtype)
        x = self.pre(c) * x_mask
        x1 = self.ssl_proj[0](x,c_mask,g)
        x2 = self.ssl_proj[1](x1)
        x3 = self.ssl_proj[2](x2,c_lengths//2,g)
        x4 = self.ssl_proj[3](x3)
        x5 = self.ssl_proj[4](x4,c_lengths//4,g)
        quantized = x5
        quantized, codes, commit_loss, quantized_list = self.quantizer(x5, layers=[0,1,2,3,4,5,6,7])
        quantized_ = self.code_emb(codes[0]).transpose(1,2)
        for i in range(10):
            noise = self.quantized_detail_enc(quantized_,c_lengths//4,g=g)
            quantized_ = quantized_ + noise
        for i in range(18):
            noise = self.quantized_detail_enc(quantized_,c_lengths//4,g=g)
            quantized_ = quantized_ + noise*5
        quantized = quantized_
        x6 = self.ssl_proj[5](quantized)
        x7 = self.ssl_proj[6](x6,c_lengths//2,g)
        x8 = self.ssl_proj[7](x7)
        x9 = self.ssl_proj[8](x8,c_lengths,g)
        x = x9
        
        f0_detail = self.f0_detail_enc(x.detach(), c_lengths, g=g)
        pred_lf0 = self.f0_decoder(x, f0_detail, x_mask, spk_emb=g)
        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        x, m_t, logs_t, _ = self.enc_text(x, x_mask, f0=f0_to_coarse(f0))
        detail = self.detail_enc(x, c_lengths, g=g)
        x = x + detail
        z_p, m_p, logs_p = self.enc_p[1](x, c_lengths,g=g)

        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noice_scale) * x_mask
        z = self.flow(z, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o,f0
    @torch.no_grad()
    def encode(self, spec, spec_lengths):
        spec_mask = torch.unsqueeze(commons.sequence_mask(spec_lengths, spec.size(2)), 1).to(spec.dtype)
        g = self.ref_enc(spec, spec_mask)
        x = self.pre(spec) * spec_mask
        x1 = self.ssl_proj[0](x,spec_mask,g)
        x2 = self.ssl_proj[1](x1)
        x3 = self.ssl_proj[2](x2,spec_lengths//2,g)
        x4 = self.ssl_proj[3](x3)
        x5 = self.ssl_proj[4](x4,spec_lengths//4,g)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x5, layers=[0,1,2,3,4,5,6,7])
        return codes[0]
    def decode(self, code, c_lengths, refer, refer_lengths, noice_scale=0.35):
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
        c_mask = torch.unsqueeze(
            commons.sequence_mask(c_lengths*4, code.size(1)*4), 1).to(refer.dtype)
        x_mask=c_mask
        g = self.ref_enc(refer, refer_mask)
        
        quantized_ = self.code_emb(code).transpose(1,2)
        for i in range(10):
            noise = self.quantized_detail_enc(quantized_,c_lengths//4,g=g)
            quantized_ = quantized_ + noise
        for i in range(18):
            noise = self.quantized_detail_enc(quantized_,c_lengths//4,g=g)
            quantized_ = quantized_ + noise*5
        quantized = quantized_
        x6 = self.ssl_proj[5](quantized)
        x7 = self.ssl_proj[6](x6,c_lengths//2,g)
        x8 = self.ssl_proj[7](x7)
        x9 = self.ssl_proj[8](x8,c_lengths,g)
        x = x9
        
        f0_detail = self.f0_detail_enc(x.detach(), c_lengths, g=g)
        pred_lf0 = self.f0_decoder(x, f0_detail, x_mask, spk_emb=g)
        f0 = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)
        
        x, m_t, logs_t, _ = self.enc_text(x, x_mask, f0=f0_to_coarse(f0))
        detail = self.detail_enc(x, c_lengths, g=g)
        x = x + detail
        z_p, m_p, logs_p = self.enc_p[1](x, c_lengths,g=g)

        z = (m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noice_scale) * x_mask
        z = self.flow(z, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g, f0=f0)
        return o
