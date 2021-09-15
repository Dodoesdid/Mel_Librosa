import librosa as l
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

# foward STFT
stft = l_out = l.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

FRAMES = stft.shape[1]

# Generate Window
hann = scipy.signal.get_window("hann", FRAME_SIZE)
#hann = hann + WINDOW_OFFSET

# Preallocate output
expected_signal_length = FRAME_SIZE + HOP_SIZE * (FRAMES - 1)
y_out = np.zeros(expected_signal_length)

for i in range(FRAMES):

    # Inverse FFT (After checking produces negative value)
    ifft = np.fft.irfft(stft[:, i]) 

    # Add offset for dividing
    #ifft = ifft + WINDOW_OFFSET

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

''' Window sum square'''
ifft_window_sum = np.zeros(expected_signal_length, dtype=y_out.dtype)

# Compute the squared window at the desired length
win_sq = scipy.signal.get_window("hann", FRAME_SIZE)

fig, axs = plt.subplots(2)
fig.suptitle('Compare')
axs[0].plot(win_sq)
axs[1].plot(win_sq ** 2)
plt.show()

win_sq = win_sq ** 2
#win_sq = util.pad_center(win_sq, n_fft)

# Overlap window sumsquare
n = len(ifft_window_sum)
n_fft = len(win_sq)
for i in range(FRAMES):
    sample = i * HOP_SIZE
    ifft_window_sum[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

# Only divide non zero elements, or ?  (np.finfo(y_out.dtype).tiny = 2.2250738585072014e-308)
approx_nonzero_indices = ifft_window_sum > np.finfo(y_out.dtype).tiny
y_out[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

# Trim data
y_out = y_out[int(n_fft // 2) : -int(n_fft // 2)]

plt.plot(y_out)
plt.title("Converted")
plt.show()

sf.write("audio_generated/mixed_istft.wav", y_out, sr)
#sf.write("audio_generate/fart.wav", y_out - y, sr)
