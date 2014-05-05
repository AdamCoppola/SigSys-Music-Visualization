from numpy import *
import scipy.io.wavfile as wio
from scipy import signal as sig
from scipy import misc

import matplotlib.pyplot as plt
import matplotlib.animation as anim

import bpm

fps = 30

# Takes an N-element array of doubles
# Returns an N/2*N matrix representing bars
# where the height of bar n is determined by element n
def imageGen(frame, Hmax):
	N = len(frame)
	frame = [x/Hmax for x in frame]

	image = zeros((N/2, N))

	for c in xrange(0, N):
		col = logspace(-2, 0, N/2)
		height = frame[c]
		col = [height if x < height else 0 for x in col]
		image[:, c] = col
	return image

def bandFFT(data, numBands, sampleRate):
	averages = empty(shape=(numBands))

	Fmax = sampleRate/2

	bandBounds = logspace(log10(20), log10((Fmax * len(data)/sampleRate)*.75), num=numBands, base=10)

	for band in range(0, len(bandBounds)-1):
		lowBound = bandBounds[band]
		highBound = bandBounds[band+1]

		avg = 0
		for i in range(int(lowBound), int(highBound)):
			avg += data[i]

		avg /= highBound - lowBound

		averages[band] = avg

	return averages

def windowAudio (data, window):
	windows = len(data)/window
	hamm = hamming(window)

	frames = zeros((windows, window))
	for i in xrange(0, windows-2):
		frames[i] = data[i*window:(i+1)*window]

	return frames

def process (data, window, rate, numBands):
	frames = windowAudio(data, window)
	hamm = hamming(window)

	spectrogram = [bandFFT(transform(x)*hamm, numBands, rate) for x in frames]

	return spectrogram

def transform (data):
	return abs(fft.fft(data))

def plotFrames (frames, frameLength, Hmax, filename):
	fig, ax = plt.subplots()

	# ax.get_xaxis().set_visible(False)
	# ax.get_yaxis().set_visible(False)

	frameImg = imageGen(frames[0], Hmax)
	img = ax.imshow(frameImg, interpolation='none', cmap='GnBu', origin='lower')

	def update_img(n):
		frameImg = imageGen(frames[n], Hmax)
		img.set_array(frameImg)

	ani = anim.FuncAnimation(fig, update_img, frames=len(frames), interval=1/float(fps))
	writer = anim.writers['ffmpeg'](fps=fps)

	ani.save(filename,writer=writer,dpi=100)

# make this better when we know which filter to use.
def freqlowpass(signal):
	filtered = signal
	cutoff = 150
	for i in xrange(0, len(signal)):
		for j in xrange(0, len(signal[i])):
			if j > cutoff:
				filtered[i][j] = 0
	return filtered

def movingAverage(data, order):
	for i in xrange(3, len(data)):
		past = 0
		averaged = list(data)
		for j in range(order):
			past = past + data[i-j]/(order + 1)
		averaged[i] = past
	return averaged

def FIRfilter(signal, rate, numSamples):
	filtered = signal
	nyquist = rate/2
	width = 5.0/nyquist
	rippleDB = 60
	N, beta = sig.kaiserord(rippleDB, width)
	cutoffHz = 20
	taps = sig.firwin(N, nyquist/2000, window=('kaiser', beta), nyq = nyquist)
	filtered = sig.lfilter(taps, 1.0, signal)
	return filtered

rate, audio = wio.read('../wav/music.wav')

seconds = float(len(audio))/rate

windowRate = fps #frames per second
windowLength = int(1/float(windowRate) * rate) #samples

smat = bpm.simMatrix(windowAudio(audio, windowLength))
# bpm.beatSpectrum(smat, seconds, 1, 0, rate)
# spec = process(audio, windowLength, rate, numBands=30)

auto = bpm.autocorr(smat)

plt.plot(linspace(0, seconds, len(auto)), auto)

plt.xlabel('Time (seconds)')
plt.ylabel('Beat Spectrum')

plt.xlim((0, seconds))

plt.title('Beat Spectrum Intensity Over Time')

plt.show()

# Hmax = amax(spec[8:-8])

# plotFrames(spec, windowLength, Hmax, 'nonfiltered.mp4')