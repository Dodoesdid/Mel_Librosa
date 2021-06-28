import Mel as m
import librosa
import matplotlib.pyplot as plt

# Load Data
y_scale , sr_scale = librosa.load('audio/scale.wav')

'''
Output in Linear
'''
# STFT
out_1 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB=False, spectro=True)

# Inverse STFT
out_6 = m.inv_STFT(out_1, frame_size=2048, hop_size=256, dB=False)

# Compare
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
librosa.display.waveshow(y_scale, sr=sr_scale, color='b', ax=ax[0])
ax[0].set(title='Original Audio', xlabel=None)
ax[0].label_outer()
librosa.display.waveshow(out_6, sr=sr_scale, color='g', ax=ax[1])
ax[1].set(title='Inverted Audio', xlabel=None)
ax[1].label_outer()
#librosa.display.waveshow(out_6 - y_scale[:len(out_6)], sr=sr_scale, color='r', ax=ax[2])
#ax[2].set(title='Difference', xlabel=None)
#ax[2].label_outer()
plt.tight_layout()
plt.show()

'''
Output in dB
'''
# STFT
out_1 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB=True, spectro=True)

# Inverse STFT
out_6 = m.inv_STFT(out_1, frame_size=2048, hop_size=256, dB=True)

# Compare
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
librosa.display.waveshow(y_scale, sr=sr_scale, color='b', ax=ax[0])
ax[0].set(title='Original Audio', xlabel=None)
ax[0].label_outer()
librosa.display.waveshow(out_6, sr=sr_scale, color='g', ax=ax[1])
ax[1].set(title='Inverted Audio', xlabel=None)
ax[1].label_outer()
#librosa.display.waveshow(out_6 - y_scale[:len(out_6)], sr=sr_scale, color='r', ax=ax[2])
#ax[2].set(title='Difference', xlabel=None)
#ax[2].label_outer()
plt.tight_layout()
plt.show()