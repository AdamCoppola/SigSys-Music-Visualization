import pylab
from numpy import *
import scipy.io.wavfile as wio
from scipy import signal, misc

import matplotlib.pyplot as plt
import matplotlib.animation as anim

fps = 30

def bandFFT(data, numBands, sampleRate):
	averages = empty(shape=(numBands))

	Fmax = sampleRate/2

	bandBounds = logspace(-2, log10((Fmax * len(data)/sampleRate)), num=numBands)

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
	hamm = hamming(window)
	for i in xrange(0, windows-2):
		fourierData = transform((data[i*window:(i+1)*window])*hamm)
		spectrogram[i] = bandFFT(fourierData, numBands, rate)

	return spectrogram

def transform (data):
	return abs(fft.fft(data))

# Plots a matrix of all the frames and saves it as a video
def plotFrames (frames, frameLength):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	x = linspace(0, len(frames[0])-1, num=len(frames[0]))

	line, = ax.plot(frames[0])
 
	fig.set_size_inches([5,5])

	def update_img(n):
		y = frames[n]
		line.set_data(x, y)
		return line,

	ani = anim.FuncAnimation(fig,update_img,frames=len(frames),interval=1/float(fps))
	writer = anim.writers['ffmpeg'](fps=fps)

	ani.save('demo.mp4',writer=writer,dpi=100)

# make this better when we know which filter to use.
def freqlowpass(signal):
	filtered = signal
	cutoff = 150
	print len(signal)
	print len(signal[1])
	for i in xrange(0, len(signal)):
		for j in xrange(0, len(signal[i])):
			if j > cutoff:
				filtered[i][j] = 0

	return filtered

rate, audio = wio.read('../wav/VVVVVV.wav')
audio, raudio = zip(*audio)

seconds = len(audio)/rate

windowRate = fps #frames per second
windowLength = int(1/float(windowRate) * rate) #samples

spec = process(audio, windowLength, rate, numBands=300)

filteredSpec = freqlowpass(spec)

plotFrames(filteredSpec, windowLength)