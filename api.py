from pypinyin import lazy_pinyin, Style
import torch
import json

MODELS = {
    'vqvae.pth':'/home/hyc/detail_tts/vqvae/logs/2024-06-25-04-45-01/model-575.pt',
}

device = 'cuda:0'
from bpe_tokenizers.voice_tokenizer import VoiceBpeTokenizer
import torch.nn.functional as F

cond_audio = '1.wav'
text = "大家好，今天来点大家想看的东西。"
# text = "霞浦县衙城镇乌旗瓦窑村水位猛涨。"
# text = '高德官方网站，拥有全面、精准的地点信息，公交驾车路线规划，特色语音导航，商家团购、优惠信息。'
# text = '四是四，十是十，十四是十四，四十是四十。'
# text = '八百标兵奔北坡，炮兵并排北边跑。炮兵怕把标兵碰，标兵怕碰炮兵炮。'
# text = '黑化肥发灰，灰化肥发黑。黑化肥挥发会发灰；灰化肥挥发会发黑。'
# text = '先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。'
text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
text = ' '+text+' '
tokenizer = VoiceBpeTokenizer('bpe_tokenizers/zh_tokenizer.json')
text_tokens = torch.IntTensor(tokenizer.encode(text)).unsqueeze(0).to(device)
text_tokens = F.pad(text_tokens, (0, 1))  # This may not be necessary.
text_tokens = text_tokens.to(device)
print(text)
print(text_tokens)
from prepare.load_infer import load_model
import torchaudio
from vqvae.utils.data_utils import spectrogram_torch,HParams
# device = 'gpu:0'
vqvae = load_model('vqvae', MODELS['vqvae.pth'], 'vqvae/configs/config.json', device)
audio,sr = torchaudio.load(cond_audio)
if audio.shape[0]>1:
    audio = audio[0].unsqueeze(0)
audio = torchaudio.transforms.Resample(sr,44100)(audio)
hps = HParams(**json.load(open('vqvae/configs/config.json')))
spec = spectrogram_torch(audio, hps.data.filter_length,
    hps.data.hop_length, hps.data.win_length, center=False).to(device)
spec_lengths = torch.LongTensor([spec.shape[-1]]).to(device)
text_lengths = torch.LongTensor([text_tokens.shape[-1]])
with torch.no_grad():
    wav = vqvae.infer(text_tokens, text_lengths, spec, spec_lengths)
torchaudio.save('gen.wav', wav.squeeze(0).cpu(), 44100)