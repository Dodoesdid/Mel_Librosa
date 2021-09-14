import vggish_mel_features as m
import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

WINDOW_OFFSET = np.finfo(np.float64).tiny
FRAME_SIZE = 1024
HOP_SIZE = 512
QUARTER_FRAME = FRAME_SIZE // 4

""" FFT test """
# test data
y, sr = sf.read("audio/mixed.wav")

plt.plot(y)
plt.title("Original Data")
plt.show()

# test numpy fft on audio
f_fft = np.fft.rfft(y)
y_fft = np.fft.irfft(f_fft)
sf.write("audio_generated/mixed_fft.wav", y_fft, sr)

''' My ISTFT '''

# frame test
frames = m.frame(y, FRAME_SIZE, HOP_SIZE)

# foward STFT
stft = m.stft_magnitude(y, FRAME_SIZE, HOP_SIZE, FRAME_SIZE)

FRAMES = stft.shape[0]

# Generate Window
hann = m.periodic_hann(FRAME_SIZE)
hann = hann + WINDOW_OFFSET

# Preallocate output
expected_signal_length = FRAME_SIZE + HOP_SIZE * (FRAMES - 1)
y_out = np.zeros(expected_signal_length)

for i in range(FRAMES):

    # Inverse FFT (After checking produces negative value)
    ifft = np.fft.irfft(stft[i, :]) 

    # Add offset for dividing
    ifft = ifft + WINDOW_OFFSET

    # remove window
    ifft = ifft * hann
    
    # Load into output
    start = i * HOP_SIZE + QUARTER_FRAME

    '''
    title = "Frame %d" % i
    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    axs[0].plot(frames[i,:])
    axs[1].plot(ifft[256 : 768])
    plt.show()
    '''

    y_out[start : (start + HOP_SIZE)] = ifft[QUARTER_FRAME : QUARTER_FRAME * 3]

plt.plot(y_out)
plt.title("Converted")
plt.show()

sf.write("audio_generated/mixed_istft.wav", y_out, sr)
#sf.write("audio_generate/fart.wav", y_out - y, sr)
