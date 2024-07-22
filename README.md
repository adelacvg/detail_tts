# Detail TTS

The model newly proposed three significant important methods to become the best practice of AR TTS.

- Although RVQ is used, the actual training employs continuous features, I call it fake discretization.
- All in one model. The model contains gpt, diffusion, vqvae, gan and flowvae all in one. One train one inference.
- Both prefixed spk emb and prompt are used to get benefit from both Valle type inference and Tortoise type training.

## Inference

check `api.py`

## Dataset prepare

Change the path contains audios in script and run

```
python prepare/0_vad_asr_save_to_jsonl.py
```

## Train and Fine Tune

```
accelerate launch train.py
```

For fine tuning, change the pretrain model load path.

## Acknowledgements

VQ and VITS from [GSV](https://github.com/RVC-Boss/GPT-SoVITS)

Diffusion and GPT from [tortoise](https://github.com/neonbjb/tortoise-tts)
