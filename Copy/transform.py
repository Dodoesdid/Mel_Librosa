from librosa.core.spectrum import stft
import Mel as m
import librosa as l
import soundfile as sf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import librosa.util as util

FRAME_SIZE = 1024
HOP_SIZE = 512

# Load Testing Data
y, sr = sf.read("audio/mixed.wav")

# Jonathan Library Test
stft_out = m.STFT(y, sr, FRAME_SIZE, HOP_SIZE)

istft_out = m.inv_STFT(stft_out, FRAME_SIZE, HOP_SIZE)
#sf.write("audio_generated/J.wav", istft_out, sr)

# Librosa library test
l_out = l.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

istft_out = l.istft(l_out, win_length=FRAME_SIZE, hop_length=HOP_SIZE)
#sf.write("istft.wav", istft_out, sr)

''' Self implementation of istft '''

# Needed functions
def get_window(window, Nx, fftbins=True):
    
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

def __overlap_add(y, ytmp, hop_length):
    # numba-accelerated overlap add for inverse stft
    # y is the pre-allocated output buffer
    # ytmp is the windowed inverse-stft frames
    # hop_length is the hop-length of the STFT analysis

    n_fft = ytmp.shape[0]
    for frame in range(ytmp.shape[1]):
        sample = frame * hop_length
        y[sample : (sample + n_fft)] += ytmp[:, frame]

def __window_ss_fill(x, win_sq, n_frames, hop_length):  # pragma: no cover
    """Helper function for window sum-square calculation."""

    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

def window_sumsquare(
    window,
    n_frames,
    hop_length=512,
    win_length=None,
    n_fft=2048,
    dtype=np.float32,
    norm=None,
):
    
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length)
    win_sq = util.normalize(win_sq, norm=norm) ** 2
    win_sq = util.pad_center(win_sq, n_fft)

    # Fill the envelope
    __window_ss_fill(x, win_sq, n_frames, hop_length)

    return x

# Get window length = FRAME_SIZE
n_fft = 2 * (l_out.shape[0] - 1)
win_length = n_fft

# Get hop length = HOP_SIZE
hop_length = HOP_SIZE

# Prepare Window
ifft_window = scipy.signal.get_window("hann", win_length)
ifft_window = util.pad_center(ifft_window, n_fft)[:, np.newaxis] #@check

# Prepare output matrix @done
n_frames = l_out.shape[1]
expected_signal_len = n_fft + hop_length * (n_frames - 1) 

dtype = util.dtype_c2r(l_out.dtype)

y = np.zeros(expected_signal_len, dtype=dtype)

# inverse FFT
n_columns = util.MAX_MEM_BLOCK // (l_out.shape[0] * l_out.itemsize)
n_columns = max(n_columns, 1)

frame = 0
for bl_s in range(0, n_frames, n_columns):
    bl_t = min(bl_s + n_columns, n_frames)

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.irfft(l_out[:, bl_s:bl_t], axis=0)

    # Overlap-add the istft block starting at the i'th frame
    __overlap_add(y[frame * hop_length :], ytmp, hop_length)

    frame += bl_t - bl_s

# Can hear voice with noise already now

# Normalize by sum of squared window SIGNIFICANTLY reduces the noise OH My !!!
'''
ifft_window_sum = window_sumsquare(
    "hann",
    n_frames,
    win_length=win_length,
    n_fft=n_fft,
    hop_length=hop_length,
    dtype=dtype,
)
'''

''' My window sumsquare '''
n = n_fft + hop_length * (n_frames - 1)
ifft_window_sum = np.zeros(n, dtype=dtype)

# Compute the squared window at the desired length
win_sq = scipy.signal.get_window("hann", win_length)
win_sq = util.normalize(win_sq, norm=None) ** 2
#win_sq = util.pad_center(win_sq, n_fft)

# Overlap window sumsquare
__window_ss_fill(ifft_window_sum, win_sq, n_frames, hop_length)

approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

# Trim data
y = y[int(n_fft // 2) : -int(n_fft // 2)]

sf.write("should.wav", y, sr)