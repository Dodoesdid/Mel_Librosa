import Mel as m
import librosa

# Load Data
y_scale , sr_scale = librosa.load('audio/scale.wav')

'''
Mel Spectrograms
   1. The human ear has limits where it can't hear frequencies to high, we need Mel scale to represent Audio 
   2. Convert f into Mel scale m = 2595 * log(1 + f/500)    
   3. We need to determine the number of mel bands in the mel scale (40 - 128)
   4. Create equally spaced mel points according to the mel bands count
   5. Convert Mel points back to frequency f = 700(10^(m/2595) - 1)
   6. Add triangle filters between each Mel point
   7. The following steps will create filter banks of (band count, frequnecy bins) dimension
   8. Appyling the dot operator on filter banks *. STFT output =>  (band count, frequnecy bins) *. (frequency bins, frame count)
      creates a Mel output dimension of (band count, frame count)
   9. This is also the problem of Mel filter banks as frequency bins usually have a bigger value than band count
      doing the Mel transformation will loose out frequency resolution resulting in data loss when inverted (my thoughts)
'''
out_5 = m.Mel(y_scale, sr_scale, frame_size=2048, hop_size=512, n_mels=128, dB=True, spectro=True)