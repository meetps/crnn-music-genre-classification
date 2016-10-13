import sys
import numpy as np
import librosa as lb

from scipy import misc

Fs         = 12000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 29.12

def log_scale_melspectrogram(path, plot=False):
    signal, sr = lb.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)

    if plot:
        melspect = melspect[np.newaxis, :]
        misc.imshow(melspect.reshape((melspect.shape[1],melspect.shape[2])))
        print(melspect.shape)

    return melspect

if __name__ == '__main__':
    log_scale_melspectrogram(sys.argv[1],True)
