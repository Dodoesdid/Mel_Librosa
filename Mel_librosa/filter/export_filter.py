import numpy as np
import librosa

SAMPLE_RATE = 16000
NFFT = 256
MEL_BINS = 64

#note: do not need to NFFT/2 because the function will do it for you
mel_filter = librosa.filters.mel(SAMPLE_RATE, NFFT, MEL_BINS)

print(mel_filter.shape)

np.savetxt("test.txt", mel_filter, "%.15f,", newline = '},\n{')