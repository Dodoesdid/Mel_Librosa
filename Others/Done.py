import numpy as np
import scipy.signal
import soundfile as sf
import matplotlib.pyplot as plt

WINDOW_OFFSET = 0.0001
FRAME_SIZE = 1024
HOP_SIZE = 512

""" FFT test """
# test data
y, sr = sf.read("audio/mixed.wav")

plt.plot(y)
plt.title("Original Audio")
plt.show()

# test numpy fft on audio
f_fft = np.fft.rfft(y)
y_fft = np.fft.irfft(f_fft)

plt.plot(y_fft)
plt.title("Converted Audio")
plt.show()

sf.write("audio_generated/mixed_fft.wav", y_fft, sr)

""" Window test """
# Generate test data
x = np.linspace(0, 10 * np.pi, 1000)
sine = np.sin(x)

plt.plot(sine)
plt.title("Original Data")
plt.show()

# Generate Window
sine_window = scipy.signal.get_window("hann", 1000, True)

# Apply window
sine_a = sine * sine_window

# Apply foward n reverse fft
sine_a = np.fft.irfft(np.fft.rfft(sine_a))

# Remove window
sine_a = sine_a * sine_window

plt.plot(sine_a)
plt.title("After Procedure")
plt.show()

plt.plot(sine - sine_a)
plt.title("Difference")
plt.show()
