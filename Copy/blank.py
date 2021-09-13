# Set the default hop, if it's not already specified
if hop_length is None:
    hop_length = int(win_length // 4)

ifft_window = get_window(window, win_length, fftbins=True)

# Pad out to match n_fft, and add a broadcasting axis
ifft_window = util.pad_center(ifft_window, n_fft)[:, np.newaxis]

//
# For efficiency, trim STFT frames according to signal length if available
if length:
    if center:
        padded_length = length + int(n_fft)
    else:
        padded_length = length
    n_frames = min(stft_matrix.shape[1], int(np.ceil(padded_length / hop_length)))
else:
    n_frames = stft_matrix.shape[1]

expected_signal_len = n_fft + hop_length * (n_frames - 1)

if dtype is None:
    dtype = util.dtype_c2r(stft_matrix.dtype)

y = np.zeros(expected_signal_len, dtype=dtype)

n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
n_columns = max(n_columns, 1)

fft = get_fftlib()

frame = 0
for bl_s in range(0, n_frames, n_columns):
    bl_t = min(bl_s + n_columns, n_frames)

    # invert the block and apply the window function
    ytmp = ifft_window * fft.irfft(stft_matrix[:, bl_s:bl_t], axis=0)

    # Overlap-add the istft block starting at the i'th frame
    __overlap_add(y[frame * hop_length :], ytmp, hop_length)

    frame += bl_t - bl_s

# Normalize by sum of squared window
ifft_window_sum = window_sumsquare(
    window,
    n_frames,
    win_length=win_length,
    n_fft=n_fft,
    hop_length=hop_length,
    dtype=dtype,
)

approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

if length is None:
    # If we don't need to control length, just do the usual center trimming
    # to eliminate padded data
    if center:
        y = y[int(n_fft // 2) : -int(n_fft // 2)]
else:
    if center:
        # If we're centering, crop off the first n_fft//2 samples
        # and then trim/pad to the target length.
        # We don't trim the end here, so that if the signal is zero-padded
        # to a longer duration, the decay is smooth by windowing
        start = int(n_fft // 2)
    else:
        # If we're not centering, start at 0 and trim/pad as necessary
        start = 0

    y = util.fix_length(y[start:], length)
