from numpy import *
import scipy.io.wavfile as wio
from scipy import signal, misc

import matplotlib.pyplot as plt

def bandFFT(data, numBands, sampleRate):
	averages = empty(shape=(numBands))

	Fmax = sampleRate/2

	bandBounds = linspace(0, Fmax * len(data)/sampleRate, num=numBands)

	for band in range(0, len(bandBounds)-1):
		lowBound = bandBounds[band]
		highBound = bandBounds[band+1]

		avg = 0
		for i in range(int(lowBound), int(highBound)):
			avg += data[i]

		avg /= highBound - lowBound

		averages[band] = avg

	return averages

def process (data, window, rate, numBands):
	windows = len(data)/window
	spectrogram = empty(shape=(windows, numBands))

	for i in range(0, windows):
		fourierData = transform(data[i*window : (i+1)*window])
		fourierData = bandFFT(fourierData, numBands, rate)

		spectrogram[i] = fourierData

	return spectrogram

def transform (data):
	return abs(fft.fft(data))

rate, audio = wio.read('../wav/VVVVVV.wav')
audio, raudio = zip(*audio)

seconds = len(audio)/rate

windowRate = 24 #frames per second
windowLength = int(1/float(windowRate) * rate) #samples

spec = process(audio, windowLength, rate, numBands=200)

print "About to pcolor"

fig = plt.figure()
ax = fig.add_subplot(111)

plt.pcolormesh(transpose(spec))

plt.xlabel('Time (Frames/Second)')

plt.ylabel('Frequency')

plt.show()