import Mel as m
import librosa

# Load Data
y_scale , sr_scale = librosa.load('audio/scale.wav')

# STFT
out_2 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB='true', spectro='true')

'''
Plotting Spectrograms
   1. This funtion will not modify the spectrogram value, it changes the y_axis representation
   2. 'linear': suitable for STFT
   3. 'log': suitable for STFT
   4. 'mel': suitable for Mel
'''
# Recreate out_2
m.spectrogram(out_2, 'linear')

# Using different y axis
m.spectrogram(out_2, 'log')
