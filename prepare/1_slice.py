import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.

from prepare.slicer2 import Slicer

audio, sr = librosa.load('example.wav', sr=None, mono=False)  # Load an audio file with librosa.
slicer = Slicer(
    sr=sr,
    threshold=-40,
    min_length=5000,
    min_interval=300,
    hop_size=10,
    max_sil_kept=500
)
chunks = slicer.slice(audio)
for i, chunk in enumerate(chunks):
    if len(chunk.shape) > 1:
        chunk = chunk.T  # Swap axes if the audio is stereo.
    soundfile.write(f'clips/example_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.