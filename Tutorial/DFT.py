import Mel as m
import librosa

# Load Data
y_sin_1k, sr_sin_1k = librosa.load('audio/sin_1k.wav')

'''
Discrerte Fourier transform - (1)
   1. Sampling frequency needs to be twice the highest f in data due to  Nyquist Frequency F(sampling) > 2 * F(data)
   2. Since we can form all signals with cosine n sine, we can think of all signals as their combination
   3. The Fourier Transform of cosine is 0.5(e^jwt + e^-jwt) which implies there will be 
      positive frequency(@wt) & negative frequency(@-wt) for all signals (w = 2 * pi * f)
   4. FT produces complex output while we need real numbers to plot a graph
   5. We will use absolute to remove the complex part but at the same time, moving the negative f to the front
      Ex. 5 + 3i & 5 - 3i (absolute) -> 16 & 34 
   6. This results in the plot below if we set the f_ratio to 1, viewing the full plot
'''
out = m.DFT(y_sin_1k, sr_sin_1k, f_ratio=1, plot=True)

'''
Discrerte Fourier transform - (2)
   1. However, this is not usually the output we need since we dont need the negative frequency
   2. Thankfullly, we have our Sampling Frequency at twice the Data Frequency
   3. All we have to do is throw away the part we dont need (0.5 ~ 1) and keep the part we need (0 ~ 0.5)
'''
out = m.DFT(y_sin_1k, sr_sin_1k, f_ratio=0.5, plot=True)
