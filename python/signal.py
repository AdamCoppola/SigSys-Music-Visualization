import pylab
from numpy import *
import scipy.io.wavfile as wio
from scipy import signal, misc

import matplotlib.pyplot as plt
import matplotlib.animation as anim

fps = 30

# Takes an N-element array of doubles
# Returns an N*N matrix representing bars
# where the height of bar n is determined by element n
def imageGen(frame, Hmax):
	N = len(frame)
	frame = [x/Hmax for x in frame]

	image = zeros((N, N))

	for c in xrange(0, N):
		col = linspace(0, 1, N)
		height = frame[c]
		col = [1 if x < height else 0 for x in col]
		image[:, c] = col
	return image

def bandFFT(data, numBands, sampleRate):
	averages = empty(shape=(numBands))

	Fmax = sampleRate/2

	bandBounds = logspace(1, log10(Fmax * len(data)/sampleRate), num=numBands, base=10)

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

def plotFrames (frames, frameLength, Hmax):
	fig, ax = plt.subplots()

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	frameImg = imageGen(frames[0], Hmax)
	img = ax.imshow(frameImg, interpolation='none', cmap='bone', origin='lower')

	def update_img(n):
		frameImg = imageGen(frames[n], Hmax)
		img.set_array(frameImg)

	ani = anim.FuncAnimation(fig, update_img, frames=len(frames), interval=1/float(fps))
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

spec = process(audio, windowLength, rate, numBands=30)
Hmax = amax(spec[8:-8])


filteredSpec = freqlowpass(spec)

plotFrames(spec, windowLength, Hmax)
