''' Window sum square'''
ifft_window_sum = np.zeros(expected_signal_length, dtype=dtype)

# Compute the squared window at the desired length
win_sq = scipy.signal.get_window("hann", win_length)
win_sq = util.normalize(win_sq, norm=None) ** 2
#win_sq = util.pad_center(win_sq, n_fft)

# Overlap window sumsquare
n = len(ifft_window_sum)
n_fft = len(win_sq)
for i in range(n_frames):
    sample = i * hop_length
    ifft_window_sum[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]


approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

# Trim data
y = y[int(n_fft // 2) : -int(n_fft // 2)]