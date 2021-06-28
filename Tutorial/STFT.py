import Mel as m
import librosa

# Load Data
y_scale , sr_scale = librosa.load('audio/scale.wav')

'''
Stort Time Fourier Transform - (1)
   1. If we want to analyze a nonperiodic signal, the best way is STFT as we can observe the frequency change through time
   2. STFT involves taking FFT multiple times in a single signal
   3. Frames - we will cut the signal into small segments using frames the frame size is the number of sampling counts taken per FFT
   4. Windowing - FFT on nonperiodic signals will create spectral leakage, we will use window functions to constraint the begining and
      ending of the signal to zero, therefore making a 'periodic signal'. Different window signals have different benefits, this may play 
      a significant role in output data. We can't choose different windows now, but will include in further versions.
   5. Hop Size - In order to make up for the loss done by window functions, we will try to overlap the frames. This is done by hopping right 
      a specific percentage of the frame size. From past experience they lie between 0.25 and 0.5
   6. Output Dimension - (frequency bins, frame count)
      frequency bins: (frame size / 2) + 1 as mentioned before in DFT, Nyquist frequency & complex numbers
      frame count: (samples - frames size) / hop size + 1 dependent on both frame & hop size
'''

out_1 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB=False, spectro=True)

'''
dB Representation
   1. The output of FT are usually complex numbers as they represent both magnitude and phase
   2. We dont need phase so we take the absolute value of FT output, creating amplitude(magnitude), valued used for above plot
   3. In Audio Processing we usually use dB to represent values, 
      so we need to turn almplitude to power, than convert it into dB
'''
out_2 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=256, dB=True, spectro=True)

'''
Tunning Output Size
   1. Increasing frame size inceases frequency resolution while decreasing time resolution, vise vera
   2. Increasing hop size decreases time resolution, vise versa 
'''
# Decrease frame size
out_3 = m.STFT(y_scale, sr_scale, frame_size=1024, hop_size=256, dB=True, spectro=False)
print('frame=2048 hop=256')
print(out_1.shape)
print('frame=1024 hop=256')
print(out_3.shape)

# Increase hop size
out_4 = m.STFT(y_scale, sr_scale, frame_size=2048, hop_size=512, dB=True, spectro=False)
print('frame=2048 hop=512')
print(out_4.shape)