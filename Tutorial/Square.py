import Mel as m
import librosa
import soundfile as sf
import numpy as np

# Load Data
y_scale , sr_scale = librosa.load('audio/voice.wav')

# STFT
out = m.STFT(y_scale, sr_scale, frame_size=1024, hop_size=128, dB='true', spectro='flase')
print(out.shape)
m.spectrogram(out, 'log')

# Append zeros
padding = np.zeros((18, 531), dtype='float64')
padding.fill(-60.)
out = np.append(out, padding, 0)
print(out.shape)
m.spectrogram(out, 'log')

# Slicing
out = out[0:513, 0:531]
print(out.shape)

# Inv STFT
out = m.inv_STFT(out, frame_size=1024, hop_size=128, dB='true')

'''
Saving time domain signal to .wav
'''
sf.write('audio_generated/inv_voice.wav', out, sr_scale)