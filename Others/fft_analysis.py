from numpy.fft import fft
from numpy.lib.utils import source
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

y, sr = sf.read("audio/mixed.wav")

def check(source):
    print(source.shape)
    print(source.dtype)
    plt.plot(source)
    plt.show()

fft_out = np.fft.fft(y)
#check(fft_out)

abs_out = np.abs(fft_out)
#check(abs_out)

rfft_out = np.fft.rfft(y)
#check(rfft_out)


