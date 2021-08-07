import Mel as m
import librosa
import soundfile as sf
import numpy as np

# Load Data
y_scale , sr_scale = librosa.load('audio/mixed.wav')

# Square
out = m.square(y_scale, frame_size=1024, hop_size=256)

# Inv Square
out = m.inv_square(out, frame_size=1024, hop_size=256)

'''
Saving time domain signal to .wav
'''
sf.write('audio_generated/inv_mixed.wav', out, sr_scale)