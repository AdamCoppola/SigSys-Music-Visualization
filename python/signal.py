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
	overlap = 1.4 # amount each window overlaps with next one (1 is no overlap)
	windows = len(data)/window
	print 'WINDOWS'
	print windows
	spectrogram = empty(shape=(windows, window/2))
	hamm = hamming(window)
	for i in xrange(0, windows-2):
		spectrogram[i]=transform((data[i*window:(i+1)*window])*hamm)
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

<<<<<<< HEAD
rate, audio = wio.read('../wav/VVVVVV.wav')
print "Done reading"
laudio, raudio = zip(*audio)
print "Done zipping"

spec = process(laudio, makeEven(rate))
plt.figure(1)
plt.plot(raudio)


plt.figure(2)
plt.pcolormesh(spec)

plt.figure(3)
=======
plt.pcolormesh(transpose(spec))

plt.xlabel('Time (Frames/Second)')

plt.ylabel('Frequency')
>>>>>>> fcd6ee9ab321e6f291db56b8f3defd60d8c6456e

plt.show()