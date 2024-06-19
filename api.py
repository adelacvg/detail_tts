from pypinyin import lazy_pinyin, Style
import torch
import json

MODELS = {
    'vqvae.pth':'/home/hyc/detail_tts/vqvae/logs/2024-06-19-07-19-25/model-194.pt',
    'gpt.pth': '/home/hyc/detail_tts/gpt/logs/2024-06-18-11-26-17/model-17.pt',
}

device = 'cuda:0'
from bpe_tokenizers.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F
cond_audio = '0.wav'
# cond_text = "霞浦县衙城镇乌旗，瓦窑村水位猛涨。"

text = "大家好，今天来点大家想看的东西。"
# text = "霞浦县衙城镇乌旗瓦窑村水位猛涨。"
# text = '高德官方网站，拥有全面、精准的地点信息，公交驾车路线规划，特色语音导航，商家团购、优惠信息。'
# text = '四是四，十是十，十四是十四，四十是四十。'
# text = '八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮。'
# text = '黑化肥发灰，灰化肥发黑。黑化肥挥发会发灰；灰化肥挥发会发黑。'
# text = '先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。诚宜开张圣听，以光先帝遗德，恢弘志士之气，不宜妄自菲薄，引喻失义，以塞忠谏之路也。'
# text = cond_text + text
pinyin = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
tokenizer = VoiceBpeTokenizer('bpe_tokenizers/zh_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(pinyin)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
print(pinyin)
print(text_tokens)
from prepare.load_infer import load_model
import torchaudio
from vqvae.utils.data_utils import spectrogram_torch,HParams
# device = 'gpu:0'
gpt = load_model('gpt',MODELS['gpt.pth'],'gpt/config.json',device)
vqvae = load_model('vqvae', MODELS['vqvae.pth'], 'vqvae/configs/config.json', device)
gpt.post_init_gpt2_config(use_deepspeed=False, kv_cache=False, half=False)
# diffusion = load_model('diffusion',MODELS['diffusion.pth'],'ttts/diffusion/config.json',device)
cond_audio = '0.wav'
audio,sr = torchaudio.load(cond_audio)
if audio.shape[0]>1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr,24000)(audio)
hps = HParams(**json.load(open('vqvae/configs/config.json')))
spec = spectrogram_torch(audio, hps.data.filter_length,
    hps.data.hop_length, hps.data.win_length, center=False).to(device)
cond_lengths = torch.LongTensor([spec.shape[-1]]).to(device)
spec_lengths = torch.LongTensor([spec.shape[-1]]).to(device)
with torch.no_grad():
    cond_vq = vqvae.encode(spec,spec_lengths)
settings = {'temperature': .8, 'length_penalty': 1.0, 'repetition_penalty': 2.0,
                    'top_p': .8,
                    'cond_free_k': 2.0, 'diffusion_temperature': 1.0}
top_p = .8
temperature = .8
autoregressive_batch_size = 1
length_penalty = 1.0
repetition_penalty = 2.0
max_mel_tokens = 600
print(spec.shape)
print(text_tokens.shape)
# text_tokens = F.pad(text_tokens,(0,400-text_tokens.shape[1]),value=0)
codes = gpt.inference_speech_tortoise(
                            spec,
                            cond_lengths,
                            text_tokens,
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            num_return_sequences=autoregressive_batch_size,
                            length_penalty=length_penalty,
                            repetition_penalty=repetition_penalty,
                            max_generate_length=max_mel_tokens
                        )
# codes = gpt.inference_speech_valle(
#                             spec,
#                             cond_lengths,
#                             text_tokens,
#                             cond_vq,
#                             do_sample=True,
#                             top_p=top_p,
#                             temperature=temperature,
#                             num_return_sequences=autoregressive_batch_size,
#                             length_penalty=length_penalty,
#                             repetition_penalty=repetition_penalty,
#                             max_generate_length=max_mel_tokens
#                         )
print(codes)
codes = codes[:,:-1]
with torch.no_grad():
    c_lengths = torch.LongTensor([codes.shape[-1]*4]).to(device)
    wav = vqvae.decode(codes, c_lengths, spec, spec_lengths)
torchaudio.save('gen.wav', wav.squeeze(0).cpu(), 44100)