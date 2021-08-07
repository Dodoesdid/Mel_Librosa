import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

NFFT = 512
PRE_EMPHASIS = 0.97
FFT_LENGTH = 256
HOP_LENGTH = 128
FRAMES = 623

'''
Load Data
'''
y, sr = sf.read("audio/mixed.wav")

# plot
plt.figure(figsize=(12, 5))
plt.title("Original")
plt.plot(y)
plt.show()

'''
Pre-Emphasis
'''
emphasized_signal = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])

# plot
plt.figure(figsize=(12, 5))
plt.title("Emphasized")
plt.plot(emphasized_signal)
plt.show()

'''
Framing
'''
