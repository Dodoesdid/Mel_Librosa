import Mel as m
import librosa
import soundfile as sf
import numpy as np

# Load Data
y_scale , sr_scale = librosa.load('audio/mixed.wav')

# STFT
out = m.STFT(y_scale, sr_scale, frame_size=1024, hop_size=256, dB=True, spectro=False)
shape = out.shape
print(shape)
frequency_bins = shape[0]
time_bins = shape[1]
m.spectrogram(out, 'log')

# Append zeros
padding = np.zeros((frequency_bins, frequency_bins - time_bins), dtype='float64')
padding.fill(-60.)
out = np.append(out, padding, 1)
print(out.shape)
m.spectrogram(out, 'log')

# Slicing
out = out[0:frequency_bins, 0:time_bins]
print(out.shape)

# Inv STFT
out = m.inv_STFT(out, frame_size=1024, hop_size=256, dB=True)

'''
Saving time domain signal to .wav
'''
sf.write('audio_generated/inv_mixed.wav', out, sr_scale)