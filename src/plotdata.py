# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:50:10 2021

@author: Mertcan
"""
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

path = "C:/Users/Mertcan/Desktop/mydata/speechdata/train/audio/dog/00f0204f_nohash_2.wav"
signal, sr = librosa.load(path)
mfcc = librosa.feature.mfcc(signal, sr, n_mfcc = 13, n_fft = 2048, hop_length = 512)

'''
plt.figure(figsize=(15, 10))

librosa.display.waveplot(signal, sr, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")
plt.colorbar()
plt.show()
'''
fft = np.fft.fft(signal)

spectrum = np.abs(fft)

f = np.linspace(0, sr, len(spectrum))

left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plot spectrum
plt.figure(figsize= (15, 10))
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")
plt.show()

'''
librosa.display.specshow(mfcc, sr=sr, hop_length=512)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.show()
'''