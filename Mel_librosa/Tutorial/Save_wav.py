import Mel as m
import librosa
import soundfile as sf

# Load Data
y_scale , sr_scale = librosa.load('audio/five_seconds.wav')

# STFT
out_1 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB=True, spectro=True)

# Inv STFT
out_6 = m.inv_STFT(out_1, frame_size=2048, hop_size=256, dB=True)

'''
Saving time domain signal to .wav
'''
sf.write('audio_generated/inv_scale.wav', out_6, sr_scale)