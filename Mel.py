import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def DFT(signal, sr, f_ratio=0.5, plot='false'):
    '''
    *** Discrete Fourier Transform ***
    Args:
        signal: time domain 1D array
        sr: sampling rate of current signal
        f_ratio: frequency axis length
        plot: 'true' if wanted to view plot
    Return:
        2D magnitude spectrum without complex numbers
    '''
    # FFT
    ft = np.fft.fft(signal)

    # Remove Complex Number
    magnitude_spectrum = np.abs(ft) ** 2

    # Plot
    if(plot == 'true'):
        plt.figure(figsize=(12, 5))
        frequency = np.linspace(0, sr, len(magnitude_spectrum))
        num_frequency_bins = int(len(frequency) * f_ratio)
        plt.plot(frequency[:num_frequency_bins], magnitude_spectrum[:num_frequency_bins])
        plt.title('Discrete Fourier Transform')
        plt.xlabel("Frequency (Hz)")
        plt.show()

    return magnitude_spectrum

def STFT(signal, sample_rate, frame_size, hop_size, dB='false', spectro='false'):
    '''
    *** Stort Time Fourier Transform ***
    Args: 
        signal: time domain 1D array
        sample_rate: sampling rate of current signal
        frame_size: number of points per FT (if bigger -> f_resolution higher, t_resolution lower)
        hop_size: suggested 0.5 ~ 0.25
        dB: if dB='true' output will be in scale of dB rather than linear
        spectro: If spectro='true', view spectrogram of data 
    Return:
        2D array of stft output y: frequency x: time
    '''
    # STFT
    Y = librosa.stft(signal, n_fft=frame_size, hop_length=hop_size)

    # Remove Complex Number
    Y = np.abs(Y)

    # Power to dB (amplitude to db ???)
    Y_dB = librosa.amplitude_to_db(Y)

    if(dB == 'true'):
        Y_out = Y_dB
    else:
        Y_out = Y

    # Visualize Spectrogram
    if (spectro == 'true'):
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(Y_out, sr=sample_rate, x_axis='time', y_axis='linear')
        plt.title('Stort Time Fourier Transform')
        plt.colorbar(format="%+2.f")
        plt.show() 

    ''' Deprecated, may delete in further versions
    # Log Ampiltude Spectrogram
    if (log == 'true'):
        Y_log = librosa.power_to_db(Y)

        plt.figure(figsize=(12, 5))
        librosa.display.specshow(Y_log, sr=sample_rate, hop_length=hop_size, x_axis='time', y_axis='log')
        plt.colorbar(format="%+2.f")
        plt.show()
    '''

    # Return
    return Y_out

def Mel(signal, sample_rate, frame_size, hop_size, n_mels, dB='false', spectro='false'):
    '''
    *** Mel Spectrogram ***
    Args:
        signal: time domain 1D array
        sample_rate: sampling rate of current signal
        frame_size: number of points per FT (if bigger -> f_resolution higher, t_resolution lower)
        hop_size: suggested 0.5 or 0.25 times of frame size
        n_mels: number of Mel filter banks, max size dependent on frame_size typical: 128
        dB: if dB='true' output will be in scale of dB rather than linear
        spectro: If spectro='true', view spectrogram of data 
    Return:
        2D array to sftt output y: Mel bands x: time
    '''

    # Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size, n_mels=n_mels)

    # Power to dB
    db_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    if(dB == 'true'):
        spectrogram =  db_mel_spectrogram
    else:
        spectrogram = mel_spectrogram

    if(spectro == 'true'):
        plt.figure(figsize=(12, 5))
        librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='mel')
        plt.title('Mel Spectrogram')
        plt.colorbar(format="%+2.f")
        plt.show()

    return spectrogram

def inv_STFT(data, frame_size, hop_size, dB='false'):
    '''
    *** Inverse STFT using Griffin Lim Algorithm ***
    Args: 
        data: 2D array STFT output
        frame_size: same as stft value
        hop_length: same as stft value
    Return:
        1D array of time domain signal
    '''

    if(dB == 'true'):
        data = librosa.db_to_amplitude(data)

    inv_data = librosa.griffinlim(data, win_length=frame_size, hop_length=hop_size)

    return inv_data

def spectrogram(data, y_axis):
    '''
    Plot 2D Spectrogram
    Args:
        data: 2D array
        sample_rate: from original data
        y_axis: 'mel', 'linear', 'log'
    '''
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(data, x_axis='time', y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.show()
    